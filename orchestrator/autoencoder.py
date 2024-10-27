import os
import numpy as np
import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
from torch.distributions import Normal

from .utils.training import kl_with_normal
from .utils.format_convert import grid2pr
from .dl_modules import PtvaeEncoder, PianoTreeDecoder, FeatDecoder, VectorQuantizerEMA


class FunctionEncoder(nn.Module):
    """Function query-net"""
    def __init__(self, emb_size=256, z_dim=128, num_channel=10):
        super(FunctionEncoder, self).__init__()
        self.cnn = nn.Sequential(nn.Conv1d(1, num_channel, kernel_size=4, stride=4, padding=0),
                                 nn.ReLU())
        self.fc = nn.Linear(num_channel * 8, emb_size)

        self.linear_mu = nn.Linear(emb_size , z_dim)
        #self.linear_var = nn.Linear(emb_size, z_dim)
        self.emb_size = emb_size
        self.z_dim = z_dim
        self.num_channel = num_channel
        self.z2hid = nn.Linear(z_dim, emb_size)
        self.hid2out = nn.Linear(emb_size, 32)
        self.mse_fn = nn.MSELoss()
        self.vq_quantizer = VectorQuantizerEMA(embedding_dim=self.num_channel, num_embeddings=128, commitment_cost=.25, decay=.9, usage_threshold=1e-9, random_start=True)
        self.batch_z = None

    def forward(self, fn, track_pad_mask):
        # fn: (bs, 32)
        bs = fn.size(0)
        fn = fn.unsqueeze(1)
        z = self.cnn(fn)#(bs, channel, 8)
        z = z.permute(0, 2, 1)#(bs, 8, channel)

        self.batch_z = z.clone()
        z, cmt_loss, perplexity = self.vq_quantizer(z, track_pad_mask.unsqueeze(1).repeat(1, 8, 1))
        z = z.reshape(bs, 8, self.num_channel).permute(0, 2, 1).reshape(bs, -1)
        
        z = self.fc(z)  # (bs, emb_size)
        z = self.linear_mu(z)

        return z, cmt_loss, perplexity
    
    def get_code_indices(self, fn):
        bs = fn.size(0)
        fn = fn.unsqueeze(1)
        z = self.cnn(fn)
        z = z.permute(0, 2, 1)
        z = self.vq_quantizer.get_code_indices(z)
        return z
    
    def infer_by_codes(self, encoding_indices):
        input_shape = encoding_indices.shape
        encoding_indices = encoding_indices.reshape(-1, 8)
        bs = encoding_indices.shape[0]
        z = self.vq_quantizer.infer_code(encoding_indices)
        z = z.reshape(bs, 8, self.num_channel).permute(0, 2, 1).reshape(bs, -1)
        z = self.fc(z)  # (bs, emb_size)
        z = self.linear_mu(z)
        z = z.reshape(*list(input_shape[:-1]), z.shape[-1])
        return z

    def decoder(self, z):
        return self.hid2out(torch.relu(self.z2hid(z)))

    def recon_loss(self, pred, fn_gt):
        return self.mse_fn(pred, fn_gt)


class Query_and_reArrange(nn.Module):
    """Q&A model for multi-track rearrangement"""
    """Zhao et al., Q&A: Query-Based Representation Learning for Multi-Track Symbolic Music re-Arrangement, in IJCAI 2023"""
    def __init__(self, name, device, trf_layers=2):
        super(Query_and_reArrange, self).__init__()

        self.name = name
        self.device = device
        
        # mixture encoder
        self.mixture_enc = PtvaeEncoder(max_simu_note=32, device=self.device, z_size=256)

        # track function encoder
        self.function_enc = FunctionEncoder(256, 256, 16)

        # feat_dec + pianotree_dec = symbolic decoder
        self.feat_dec = FeatDecoder(z_dim=256)  # for key feature reconstruction
        self.feat_emb_layer = nn.Linear(3, 64)
        self.pianotree_dec = PianoTreeDecoder(z_size=256, feat_emb_dim=64, device=device)

        self.Transformer_layers = nn.ModuleDict({})
        self.trf_layers = trf_layers
        for idx in range(self.trf_layers):
            self.Transformer_layers[f'layer_{idx}'] = TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, dropout=.1, activation=F.gelu, batch_first=True)

        self.prog_embedding = nn.Embedding(num_embeddings=35, embedding_dim=256, padding_idx=34)

        self.trf_mu = nn.Linear(256, 256)
        self.trf_var = nn.Linear(256, 256)

    def run(self, pno_red, program, function, pno_tree=None, feat=None, track_pad_mask=None, tfr1=0, tfr2=0, inference=False, mel_id=None):
        """
        Forward path of the model in training (w/o computing loss).
        """

        batch, track, time = function.shape
        max_simu_note = 16  # 14 + sos + eos
        
        dist_pn, _, _ = self.mixture_enc(pno_red) 
        if inference:
            z_pn = dist_pn.mean
        else:
            z_pn = dist_pn.rsample()
        if track_pad_mask is None:
            track_pad_mask = torch.zeros(batch, track, dtype=bool).to(z_pn.device)

        function = function.reshape(-1, 32)
        z_fn, cmt_loss, plty = self.function_enc(function, track_pad_mask)
        function_recon = self.function_enc.decoder(z_fn).reshape(batch, track, -1)

        z_fn = z_fn.reshape(batch, track, -1)   #(batch, track, 256),
        z = torch.cat([
                    z_pn.unsqueeze(1), #(batch, 1, 256)
                    z_fn + self.prog_embedding(program)],
                    dim=1)  #z: (batch, track+1, 256)

        if not inference:
            trf_mask = torch.cat([torch.zeros(batch, 1, device=z.device).bool(), track_pad_mask], dim=-1)   #(batch, track+1)
        else:
            trf_mask = torch.zeros(batch, track+1, device=z.device).bool()

        for idx in range(self.trf_layers):
            z = self.Transformer_layers[f'layer_{idx}'](src=z, src_key_padding_mask=trf_mask)


        z = z[:, 1:].reshape(-1, 256)
        mu = self.trf_mu(z)
        var = self.trf_var(z).exp_()

        dist_trf = Normal(mu, var)
        if inference and (mel_id is None):
            z = dist_trf.mean
        elif inference and (mel_id is not None):
            z1 = dist_trf.mean.reshape(batch, track, 256)
            z2 = dist_trf.rsample().reshape(batch, track, 256)
            z = torch.cat([z1[:, :mel_id], z2[:, mel_id: mel_id+1], z1[:, mel_id+1:]], dim=1).reshape(-1, 256)
        else:
            z = dist_trf.rsample()

        if not inference:
            feat = feat.reshape(-1, time, 3)
        #reconstruct key feature for self-supervision during training
        recon_feat = self.feat_dec(z, inference, tfr1, feat)    #(batch*track, time, 3)
        #embed the reconstructed feature (without applying argmax)
        feat_emb = self.feat_emb_layer(recon_feat)

        #prepare the teacher-forcing data for pianotree decoder
        if inference:
            embedded_pno_tree = None
            pno_tree_lgths = None
        else:
            embedded_pno_tree, pno_tree_lgths = self.pianotree_dec.emb_x(pno_tree.reshape(-1, time, max_simu_note, 6))

        #pianotree decoder
        recon_pitch, recon_dur = \
            self.pianotree_dec(z, inference, embedded_pno_tree, pno_tree_lgths, tfr1, tfr2, feat_emb)

        recon_pitch = recon_pitch.reshape(batch, track, time, max_simu_note-1, 130)
        recon_dur = recon_dur.reshape(batch, track, time, max_simu_note-1, 5, 2)
        recon_feat = recon_feat.reshape(batch, track, time, 3)

        return recon_pitch, recon_dur, recon_feat, \
               function_recon, \
               dist_pn, dist_trf, \
               cmt_loss, plty

    def loss_calc(self, pno_tree, feat, function, 
                      recon_pitch, recon_dur, recon_feat, function_recon,
                      dist_pn, dist_trf, cmt_loss, plty, track_pad_mask,
                      beta, weights):
        """ Compute the loss from ground truth and the output of self.run()"""
        mask = torch.logical_not(track_pad_mask)
        # pianotree recon loss
        pno_tree_l, pitch_l, dur_l = \
            self.pianotree_dec.recon_loss(pno_tree[mask], 
                                          recon_pitch[mask], 
                                          recon_dur[mask],
                                          weights)
        # key feature reconstruction loss
        feat_l, onset_feat_l, int_feat_l, center_feat_l = \
            self.feat_dec.recon_loss(feat[mask], recon_feat[mask])

        func_l = self.function_enc.recon_loss(function_recon[mask], function[mask])
        vqvae_l = func_l + cmt_loss

        # kl losses
        kl_pn = kl_with_normal(dist_pn)
        kl_trf = kl_with_normal(dist_trf)

        kl_l = beta * (kl_pn + kl_trf)

        loss = pno_tree_l + feat_l + kl_l + vqvae_l

        return loss, pno_tree_l, pitch_l, dur_l, \
               kl_l, kl_pn, kl_trf, \
               feat_l, onset_feat_l, int_feat_l, center_feat_l, \
               vqvae_l, func_l, cmt_loss, plty

    def loss(self, pno_red, progam, function, pno_tree, feat, track_pad_mask, tfr1, tfr2,
             beta=0.01, weights=(1, 0.5)):
        """forward and calculate loss"""
        output = self.run(pno_red, progam, function, pno_tree, feat, track_pad_mask, tfr1, tfr2)
        return self.loss_calc(pno_tree, feat, function, *output, track_pad_mask, beta, weights)
    
    def output_process(self, recon_pitch, recon_dur):
        grid_recon = torch.cat([recon_pitch.max(-1)[-1].unsqueeze(-1), recon_dur.max(-1)[-1]], dim=-1)
        _, track, _, max_simu_note, grid_dim = grid_recon.shape
        grid_recon = grid_recon.permute(1, 0, 2, 3, 4)
        grid_recon = grid_recon.reshape(track, -1, max_simu_note, grid_dim)
        pr_recon = np.array([grid2pr(matrix) for matrix in grid_recon.detach().cpu().numpy()])
        return pr_recon

    def inference(self, pno_red, program, function, mel_id=None):
        self.eval()
        with torch.no_grad():
            recon_pitch, recon_dur, _, _, _, _, _, _ = self.run(pno_red, program, function, inference=True, mel_id=mel_id)
            pr_recon = self.output_process(recon_pitch, recon_dur)
        return pr_recon
    
    def infer_with_function_codes(self, z_pn, program, z_fn):
        #z_pn: (batch, num2bar, 256)
        #program: (batch, 1, track)
        #z_fn: (batch, num2bar, track, 128)

        z = torch.cat([ z_pn.unsqueeze(2), #(batch, num2bar, 1, 256)
                        z_fn + self.prog_embedding(program)],
                    dim=2)  #z: (batch, num2bar, track+1, 256)"""
        
        z = z.reshape(-1, *list(z.shape[2:]))
        for idx in range(self.trf_layers):
            z = self.Transformer_layers[f'layer_{idx}'](src=z)
        
        z = z[:, 1:].reshape(-1, 256)

        mu = self.trf_mu(z)
        var = self.trf_var(z).exp_()
        dist_trf = Normal(mu, var)
        z = dist_trf.mean

        recon_feat = self.feat_dec(z, True, 0, None)
        feat_emb = self.feat_emb_layer(recon_feat)

        # pianotree decoder
        recon_pitch, recon_dur = \
            self.pianotree_dec(z, True, None, None, 0, 0, feat_emb)

        recon_pitch = recon_pitch.reshape(*list(z_fn.shape[:3]), 32, 15, 130)
        recon_dur = recon_dur.reshape(*list(z_fn.shape[:3]), 32, 15, 5, 2)
        return recon_pitch, recon_dur

    def forward(self, mode, *input, **kwargs):
        if mode in ["run", 0]:
            return self.run(*input, **kwargs)
        elif mode in ['loss', 'train', 1]:
            return self.loss(*input, **kwargs)
        elif mode in ['inference', 'eval', 'val', 2]:
            return self.inference(*input, **kwargs)
        else:
            raise NotImplementedError



if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES']= '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    from autoencoder_dataset import Slakh2100_Pop909_Dataset, collate_fn
    from torch.utils.data import DataLoader

    DEVICE = 'cuda:0'
    BATCH_SIZE = 4

    model = Query_and_reArrange(name='debug', trf_layers=2, device=DEVICE)
    model.load_state_dict(torch.load("/data1/zhaojw/data_file_dir/params_autoencoder.pt", map_location=torch.device('cpu')), strict=True)
    model.to(DEVICE)

    slakh_dir = "/data1/zhaojw/data_file_dir/Slakh2100_inference_set/"
    pop909_dir = None
    val_set = Slakh2100_Pop909_Dataset(slakh_dir, pop909_dir, debug_mode=True, split='validation', mode='train')
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, DEVICE, pitch_shift=False))

    for idx, batch in enumerate(val_loader):
        losses = model('loss', *batch, tfr1=0, tfr2=0, beta=0.01, weights=(1, 0.5))
        print(losses)
        break
 