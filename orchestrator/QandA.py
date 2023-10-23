import os
from torch import nn
from .utils import kl_with_normal
import torch
from .dl_modules import PtvaeEncoder, PianoTreeDecoder, TextureEncoder, AdaptFeatDecoder, VectorQuantizerEMA, VectorQuantizer

from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
from torch.distributions import Normal


class FuncPitchEncoder(nn.Module):
    def __init__(self, emb_size=256, z_dim=128, num_channel=10):
        super(FuncPitchEncoder, self).__init__()
        self.cnn = nn.Sequential(nn.Conv1d(1, num_channel, kernel_size=12,
                                           stride=1, padding=0),
                                 nn.ReLU(),
                                 nn.MaxPool1d(kernel_size=4, stride=4))
        self.fc = nn.Linear(num_channel * 29, emb_size)
        self.linear_mu = nn.Linear(emb_size, z_dim)
        #self.linear_var = nn.Linear(emb_size, z_dim)
        self.emb_size = emb_size
        self.z_dim = z_dim
        self.z2hid = nn.Linear(z_dim, emb_size)
        self.hid2out = nn.Linear(emb_size, 128)
        self.mse_func = nn.MSELoss()
        self.vq_quantizer = VectorQuantizerEMA(embedding_dim=z_dim, num_embeddings=64, commitment_cost=.25, decay=.9, usage_threshold=1e-9, random_start=True)
        #self.vq_quantizer = VectorQuantizer(embedding_dim=z_dim, num_embeddings=256, commitment_cost=.25, usage_threshold=1e-9, random_start=True)
        self.batch_z = None

    def forward(self, pr, track_pad_mask):
        # pr: (bs, 128)
        bs = pr.size(0)
        pr = pr.unsqueeze(1)
        pr = self.cnn(pr).reshape(bs, -1)
        pr = self.fc(pr)  # (bs, emb_size)
        mu = self.linear_mu(pr)
        self.batch_z = mu.clone()
        z, cmt_loss, perplexity = self.vq_quantizer(mu, track_pad_mask)
        return z, cmt_loss, perplexity
    
    def get_code_indices(self, pr):
        bs = pr.size(0)
        pr = pr.unsqueeze(1)
        pr = self.cnn(pr).reshape(bs, -1)
        pr = self.fc(pr)  # (bs, emb_size)
        pr = self.linear_mu(pr)
        pr = self.vq_quantizer.get_code_indices(pr)
        return pr
    
    def infer_by_codes(self, encoding_indices):
        z = self.vq_quantizer.infer_code(encoding_indices)
        return z

    def decoder(self, z):
        return self.hid2out(torch.relu(self.z2hid(z)))

    def recon_loss(self, pred, func_gt):
        return self.mse_func(pred, func_gt)


class FuncTimeEncoder(nn.Module):
    def __init__(self, emb_size=256, z_dim=128, num_channel=10):
        super(FuncTimeEncoder, self).__init__()
        self.cnn = nn.Sequential(nn.Conv1d(1, num_channel, kernel_size=4,
                                           stride=4, padding=0),
                                 nn.ReLU())
        self.fc = nn.Linear(num_channel * 8, emb_size)

        self.linear_mu = nn.Linear(emb_size , z_dim)
        self.emb_size = emb_size
        self.z_dim = z_dim
        self.num_channel = num_channel
        self.z2hid = nn.Linear(z_dim, emb_size)
        self.hid2out = nn.Linear(emb_size, 32)
        self.mse_func = nn.MSELoss()
        self.vq_quantizer = VectorQuantizerEMA(embedding_dim=(self.num_channel*8)//8, num_embeddings=128, commitment_cost=.25, decay=.9, usage_threshold=1e-9, random_start=True)
        #self.vq_quantizer = VectorQuantizer(embedding_dim=(self.num_channel*8)//4, num_embeddings=256, commitment_cost=.25, usage_threshold=1e-9, random_start=True)
        self.batch_z = None

    def forward(self, pr, track_pad_mask):
        # pr: (bs, 32)
        bs = pr.size(0)
        pr = pr.unsqueeze(1)
        pr = self.cnn(pr)#.reshape(bs, -1)  #(bs, channel, 8)
        pr = pr.permute(0, 2, 1).reshape(bs, -1)
        z = pr.reshape(bs, 8, (self.num_channel*8)//8)
        self.batch_z = z.clone()
        z, cmt_loss, perplexity = self.vq_quantizer(z, track_pad_mask.unsqueeze(1).repeat(1, 8, 1))
        z = z.reshape(bs, 8, self.num_channel).permute(0, 2, 1).reshape(bs, -1)
        
        z = self.fc(z)  # (bs, emb_size)
        z = self.linear_mu(z)
        return z, cmt_loss, perplexity
    
    def get_code_indices(self, pr):
        bs = pr.size(0)
        pr = pr.unsqueeze(1)
        pr = self.cnn(pr)
        pr = pr.permute(0, 2, 1).reshape(bs, -1)
        pr = pr.reshape(bs, 8, (self.num_channel*8)//8)
        pr = self.vq_quantizer.get_code_indices(pr)
        return pr.reshape(bs, 8)
    
    def infer_by_codes(self, encoding_indices):
        #print('encoding_indices', encoding_indices.shape)
        input_shape = encoding_indices.shape
        encoding_indices = encoding_indices.reshape(-1, 8)
        bs = encoding_indices.shape[0]
        z = self.vq_quantizer.infer_code(encoding_indices)
        #print('z', z.shape)
        z = z.reshape(bs, 8, self.num_channel).permute(0, 2, 1).reshape(bs, -1)
        z = self.fc(z)  # (bs, emb_size)
        z = self.linear_mu(z)
        z = z.reshape(*list(input_shape[:-1]), z.shape[-1])
        #print(z.shape)
        return z

    def decoder(self, z):
        return self.hid2out(torch.relu(self.z2hid(z)))

    def recon_loss(self, pred, func_gt):
        return self.mse_func(pred, func_gt)



class QandA(nn.Module):
    def __init__(self, name, device,
                 trf_layers=1,
                 stage=0):
        super(QandA, self).__init__()

        self.name = name
        self.device = device

        # symbolic encoder
        self.prmat_enc_fltn = PtvaeEncoder(max_simu_note=32, device=self.device, z_size=256)

        # track function encoder
        self.func_pitch_enc = FuncPitchEncoder(256, 128, 16)
        self.func_time_enc = FuncTimeEncoder(256, 128, 16)

        # feat_dec + pianotree_dec = symbolic decoder
        self.feat_dec = AdaptFeatDecoder(z_dim=256)  # for symbolic feature recon
        self.feat_emb_layer = nn.Linear(3, 64)
        self.pianotree_dec = PianoTreeDecoder(z_size=256, feat_emb_dim=64, device=device)

        self.Transformer_layers = nn.ModuleDict({})
        self.trf_layers = trf_layers
        for idx in range(self.trf_layers):
            self.Transformer_layers[f'layer_{idx}'] = TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, dropout=.1, activation=F.gelu, batch_first=True)

        self.prog_embedding = nn.Embedding(num_embeddings=35, embedding_dim=256, padding_idx=34)

        self.eq_feat_head = nn.Linear(256, 4)

        self.trf_mu = nn.Linear(256, 256)
        self.trf_var = nn.Linear(256, 256)

        self.stage = stage

    @property
    def z_chd_dim(self):
        return self.chord_enc.z_dim

    @property
    def z_aud_dim(self):
        return self.frame_enc.z_dim

    @property
    def z_sym_dim(self):
        return self.prmat_enc.z_dim

    
    def run(self, pno_tree, pno_tree_fltn, prog, feat, track_pad_mask, func_pitch, func_time, tfr1=0, tfr2=0, tfr3=0, inference=False, sample_melody=False):
        """
        Forward path of the model in training (w/o computing loss).
        """
        #pno_tree: (batch, max_track, time, max_simu_note, 6)
        #chd: (batch, time', 36)
        #pr_fltn: (batch, max_track, time, 128)
        #prog: (batch, 5, max_track)
        #track_pad_mask: (batch, max_track)
        #feat: (batch, max_track, time, 3)
        #func_pitch: (batch, max_track, 128)
        #func_time: (batch, max_track, 32)


        if inference:
            batch, track = track_pad_mask.shape
            _, time, _, _ = pno_tree_fltn.shape
            max_simu_note = 16
        else:
            batch, track, time, max_simu_note, _ = pno_tree.shape
        #print('pno_tree', pno_tree.shape)

        dist_sym, _, _ = self.prmat_enc_fltn(pno_tree_fltn)   #
        if inference:
            z_sym = dist_sym.mean
        else:
            z_sym = dist_sym.rsample()


        # compute symbolic-texture representation
        if self.stage in [0, 1, 3]:
            #print('pr_mat', pr_mat.shape)
            func_pitch = func_pitch.reshape(-1, 128)
            z_fp, cmt_loss_p, plty_p = self.func_pitch_enc(func_pitch, track_pad_mask)

            func_time = func_time.reshape(-1, 32)
            z_ft, cmt_loss_t, plty_t = self.func_time_enc(func_time, track_pad_mask)

            fp_recon = self.func_pitch_enc.decoder(z_fp).reshape(batch, track, -1)
            ft_recon = self.func_time_enc.decoder(z_ft).reshape(batch, track, -1)

            z_func = torch.cat([
                                z_fp.reshape(batch, track, -1),
                                z_ft.reshape(batch, track, -1)
                                ],
                                dim=-1) #(batch, track, 256),
        else:  # self.stage == 2 (fine-tuning stage), dist_sym abandoned.
            #TODO
            pass

        #print('prog', prog.shape)
        #print('prog embedding', self.prog_embedding(prog[:, 0]).shape)

        z = torch.cat([
                        z_sym.unsqueeze(1), #(batch, 1, 256)
                        z_func + self.prog_embedding(prog)],
                    dim=1)  #z: (batch, track+1, 256)"""


        trf_mask = torch.cat([torch.zeros(batch, 1, device=z.device).bool(), track_pad_mask], dim=-1)   #(batch, track+1)
        for idx in range(self.trf_layers):
            z = self.Transformer_layers[f'layer_{idx}'](src=z, src_key_padding_mask=trf_mask)

        # reconstruct symbolic feature using audio-texture repr.
        z = z[:, 1:].reshape(-1, 256)

        #eq_feat = self.eq_feat_head(z).reshape(batch, track, 4)

        mu = self.trf_mu(z)
        var = self.trf_var(z).exp_()

        #eq_feat = self.eq_feat_head(mu).reshape(batch, track, 4)

        dist_trf = Normal(mu, var)
        if inference and (not sample_melody):
            z = dist_trf.mean
        elif inference and sample_melody:
            z1 = dist_trf.mean.reshape(batch, track, 256)
            z2 = dist_trf.rsample().reshape(batch, track, 256)
            z = torch.cat([z2[:, 0: 1], z1[:, 1:]], dim=1).reshape(-1, 256)
        else:
            z = dist_trf.rsample()
        #z = z.reshape(batch, track, 256)

        if not inference:
            feat = feat.reshape(-1, time, 3)
        recon_feat = self.feat_dec(z, inference, tfr1, feat)    #(batch*track, time, 3)
        # embed the reconstructed feature (without applying argmax)
        feat_emb = self.feat_emb_layer(recon_feat)

        # prepare the teacher-forcing data for pianotree decoder
        if inference:
            embedded_pno_tree = None
            pno_tree_lgths = None
        else:
            embedded_pno_tree, pno_tree_lgths = self.pianotree_dec.emb_x(pno_tree.reshape(-1, time, max_simu_note, 6))

        # pianotree decoder
        recon_pitch, recon_dur = \
            self.pianotree_dec(z, inference, embedded_pno_tree, pno_tree_lgths, tfr1, tfr2, feat_emb)

        recon_pitch = recon_pitch.reshape(batch, track, time, max_simu_note-1, 130)
        recon_dur = recon_dur.reshape(batch, track, time, max_simu_note-1, 5, 2)
        recon_feat = recon_feat.reshape(batch, track, time, 3)

        return recon_pitch, recon_dur, recon_feat, \
               fp_recon, ft_recon, \
               dist_sym, dist_trf, \
               cmt_loss_p, plty_p, \
               cmt_loss_t, plty_t


    def loss_function(self, pno_tree, feat, func_pitch, func_time, recon_pitch, recon_dur, recon_feat,
                      fp_recon, ft_recon,
                      dist_sym, dist_trf, cmt_loss_p, plty_p, cmt_loss_t, plty_t, track_pad_mask,
                      beta, weights):
        """ Compute the loss from ground truth and the output of self.run()"""
        mask = torch.logical_not(track_pad_mask)
        # pianotree recon loss
        pno_tree_l, pitch_l, dur_l = \
            self.pianotree_dec.recon_loss(pno_tree[mask], 
                                          recon_pitch[mask], 
                                          recon_dur[mask],
                                          weights, False)

        # feature prediction loss
        feat_l, onset_feat_l, int_feat_l, center_feat_l = \
            self.feat_dec.recon_loss(feat[mask], recon_feat[mask])

        fp_l = self.func_pitch_enc.recon_loss(fp_recon[mask], func_pitch[mask])
        ft_l = self.func_time_enc.recon_loss(ft_recon[mask], func_time[mask])
        func_l = (fp_l + cmt_loss_p) + (ft_l + cmt_loss_t)

        # kl losses
        kl_sym = kl_with_normal(dist_sym)
        kl_trf = kl_with_normal(dist_trf)

        if self.stage == 0:
            # beta keeps annealing from 0 - 0.01
            kl_l = beta * (kl_sym + kl_trf)
        else:  # self.stage == 3
            # autoregressive fine-tuning
            pass

        loss = pno_tree_l + feat_l + kl_l + func_l

        return loss, pno_tree_l, pitch_l, dur_l, \
               kl_l, kl_sym, kl_trf, \
               feat_l, onset_feat_l, int_feat_l, center_feat_l, \
               func_l, fp_l, ft_l, cmt_loss_p, cmt_loss_t, plty_p, plty_t


    def loss(self, pno_tree, pno_tree_fltn, prog, feat, track_pad_mask, func_pitch_batch, func_time_batch, tfr1, tfr2, tfr3,
             beta=0.01, weights=(1, 0.5)):
        """
        Forward path during training with loss computation.
        :param pno_tree: (B, track, 32, 16, 6) ground truth for teacher forcing
        :param chd: (B, 8, 36) chord input
        :param spec: (B, 229, 153) audio input. Log mel-spectrogram. (n_mels=229)
        :param pr_mat: (B, track, 32, 128) (with proper corruption) symbolic input.
        :param prog: (B, 5, track), track program and feature for embedding
        :param feat: (B, track, 32, 3) ground truth for teacher forcing
        :param track_pad_mask: (B, track), pad mask for Transformer. BoolTensor, with True indicating mask
        :param tfr1: teacher forcing ratio 1 (1st-hierarchy RNNs except chord)
        :param tfr2: teacher forcing ratio 2 (2nd-hierarchy RNNs except chord)
        :param tfr3: teacher forcing ratio 3 (for chord decoder)
        :param beta: kl annealing parameter
        :param weights: weighting parameter for pitch and dur in PianoTree.
        :return: losses (first argument is the total loss.)
        """

        output = self.run(pno_tree, pno_tree_fltn, prog, feat, track_pad_mask, func_pitch_batch, func_time_batch, tfr1, tfr2, tfr3)

        return self.loss_function(pno_tree, feat, func_pitch_batch, func_time_batch, *output, track_pad_mask, beta, weights)


    def forward(self, mode, *input, **kwargs):
        if mode in ["run", 0]:
            return self.run(*input, **kwargs)
        elif mode in ['loss', 'train', 1]:
            return self.loss(*input, **kwargs)
        elif mode in ['inference', 'eval', 'val', 2]:
            return self.inference(*input, **kwargs)
        else:
            raise NotImplementedError

    def load_model(self, model_path, map_location=None):
        if map_location is None:
            map_location = self.device
        dic = torch.load(model_path, map_location=map_location)
        for name in list(dic.keys()):
            dic[name.replace('module.', '')] = dic.pop(name)
        self.load_state_dict(dic)
        self.to(self.device)

    def infer_with_function_codes(self, z_sym, prog, z_fp, z_ft):
        #z_sym: (batch, 256)
        #prog: (batch, track)
        #z_fp: (batch, track, 128)
        #z_fp: (batch, track, 128)

        z_func = torch.cat([z_fp, z_ft], dim=-1) 
        z = torch.cat([ z_sym.unsqueeze(1), #(batch, 1, 256)
                        z_func + self.prog_embedding(prog)],
                    dim=1)  #z: (batch, track+1, 256)"""
        
        for idx in range(self.trf_layers):
            z = self.Transformer_layers[f'layer_{idx}'](src=z)
        
        z = z[:, 1:].reshape(-1, 256)

        mu = self.trf_mu(z)
        var = self.trf_var(z).exp_()
        dist_trf = Normal(mu, var)
        z = dist_trf.mean

        recon_feat = self.feat_dec(z, True, 0, None)
        feat_emb = self.feat_emb_layer(recon_feat)

        # prepare the teacher-forcing data for pianotree decoder
        embedded_pno_tree = None
        pno_tree_lgths = None
    
        # pianotree decoder
        recon_pitch, recon_dur = \
            self.pianotree_dec(z, True, embedded_pno_tree, pno_tree_lgths, 0, 0, feat_emb)

        recon_pitch = recon_pitch.reshape(*list(prog.shape), 32, 15, 130)
        recon_dur = recon_dur.reshape(*list(prog.shape), 32, 15, 5, 2)
        return recon_pitch, recon_dur


    def inference(self, audio, chord, sym_prompt=None):
        """
        Forward path during inference. By default, symbolic source is not used.
        :param audio: (B, 229, 153) audio input.
            Log mel-spectrogram. (n_mels=229)
        :param chord: (B, 8, 36) chord input
        :param sym_prompt: (B, 32, 128) symbolic prompt.
            By default, None.
        :return: pianotree prediction (B, 32, 15, 6) numpy array.
        """

        self.eval()
        with torch.no_grad():
            z_chd = self.chord_enc(chord).mean
            z_aud = self.audio_enc(audio).mean

            z_sym = \
                torch.zeros(z_aud.size(0), self.z_sym_dim,
                            dtype=z_aud.dtype, device=z_aud.device) \
                if sym_prompt is None else self.prmat_enc(sym_prompt).mean

            z = torch.cat([z_chd, z_aud, z_sym], -1)

            recon_feat = self.feat_dec(z_aud, True, 0., None)
            feat_emb = self.feat_emb_layer(recon_feat)
            recon_pitch, recon_dur = \
                self.pianotree_dec(z, True, None, None, 0., 0., feat_emb)

        # convert to (argmax) pianotree format, numpy array.
        pred = self.pianotree_dec.output_to_numpy(recon_pitch.cpu(),
                                                  recon_dur.cpu())[0]
        return pred
