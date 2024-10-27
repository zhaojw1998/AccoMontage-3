import math
import random
import torch
from torch import nn
import torch.nn.functional as F
from .autoencoder import Query_and_reArrange
from .TransformerEncoderLayer import TransformerEncoderLayer as TransformerEncoderLayerRPE
from .prior_dataset import NUM_INSTR_CLASS, NUM_TIME_CODE, TOTAL_LEN_BIN, ABS_POS_BIN, REL_POS_BIN

class Prior(nn.Module):
    def __init__(self, piano_encoder=None,
                       function_encoder=None,
                       context_enc_layer=12, 
                       function_dec_layer=12, 
                       d_model=256, 
                       nhead=8, 
                       dim_feedforward=1024, 
                       dropout=.1, 
                       inference=False,
                       autoencoder=None,
                       DEVICE='cuda:0'):
        super(Prior, self).__init__()

        # embeddings
        self.func_embedding = nn.Embedding(num_embeddings=NUM_TIME_CODE+1, embedding_dim=d_model, padding_idx=NUM_TIME_CODE)
        self.prog_embedding = nn.Embedding(num_embeddings=NUM_INSTR_CLASS+1, embedding_dim=d_model, padding_idx=NUM_INSTR_CLASS)
        self.total_len_embedding = nn.Embedding(num_embeddings=len(TOTAL_LEN_BIN)+1, embedding_dim=d_model, padding_idx=len(TOTAL_LEN_BIN))
        self.abs_pos_embedding = nn.Embedding(num_embeddings=len(ABS_POS_BIN)+1, embedding_dim=d_model, padding_idx=len(ABS_POS_BIN))
        self.rel_pos_embedding = nn.Embedding(num_embeddings=len(REL_POS_BIN)+1, embedding_dim=d_model, padding_idx=len(REL_POS_BIN))

        self.start_embedding = nn.Parameter(torch.empty(NUM_INSTR_CLASS+1, d_model))
        nn.init.normal_(self.start_embedding)
        with torch.no_grad():
                self.start_embedding[NUM_INSTR_CLASS].fill_(0)

        #pre-trained encoders
        if not inference:
            self.piano_encoder = piano_encoder
            for param in self.piano_encoder.parameters():
                param.requires_grad = False
            self.function_encoder = function_encoder
            for param in self.function_encoder.parameters():
                param.requires_grad = False
        else:
            self.autoencoder = autoencoder
            self.piano_encoder = self.autoencoder.mixture_enc
            self.function_encoder = self.autoencoder.function_enc

        self.context_enc = nn.TransformerEncoder(
                                nn.TransformerEncoderLayer(d_model=d_model, 
                                                            nhead=nhead, 
                                                            dim_feedforward=dim_feedforward, 
                                                            dropout=dropout, 
                                                            activation=F.gelu, 
                                                            batch_first=True, 
                                                            norm_first=True,
                                                            device=DEVICE),
                                num_layers=context_enc_layer)
        #multi-track Transformer
        self.mt_trf = nn.ModuleDict({})
        for layer in range(function_dec_layer):
            self.mt_trf[f'track_layer_{layer}'] = TransformerEncoderLayerRPE(d_model=d_model, 
                                                            nhead=nhead, 
                                                            dim_feedforward=dim_feedforward, 
                                                            dropout=dropout, 
                                                            norm_first=True,
                                                            max_len=18).to(DEVICE)
            self.mt_trf[f'time_layer_{layer}'] = nn.TransformerDecoderLayer(d_model=d_model, 
                                                            nhead=nhead, 
                                                            dim_feedforward=dim_feedforward, 
                                                            dropout=dropout, 
                                                            activation=F.gelu, 
                                                            batch_first=True, 
                                                            norm_first=True,
                                                            device=DEVICE)
        
        #positional encoding
        self.max_len = 1000
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, self.max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        pe = pe.to(DEVICE)
        self.register_buffer('pe', pe)
        
        #decoder output module 
        self.func_out_linear = nn.Linear(d_model, NUM_TIME_CODE)

        #constants
        self.d_model = d_model
        self.function_dec_layer = function_dec_layer

        #loss function
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def generate_square_subsequent_mask(self, sz=15):
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

    def fn_get_next_token(self, token, gt=None):
        #token: (batch, codebook_size)
        #gt: (bs,)
        if gt is None:
            idx = token.max(-1)[1]
        else:
            idx = gt
        token = torch.zeros_like(token, device=token.device)
        arange = torch.arange(token.shape[0], device=token.device).long()
        token[arange, idx] = 1
        return token.unsqueeze(1)   #one-hot shape (batch, 1, fn_codebook_size)

    def run(self, z_pn, prog, fn, tm_mask, tk_mask, total_len, abs_pos, rel_pos):
        #z_pn: (batch, n_segments, 256) latent content representation
        #prog: (batch, max_track)
        #function: (batch, n_segments, max_track, 8) latent style representation
        #tm_mask: (batch, n_segments)
        #tk_mask: (batch, n_segments)
        #total_len: (batch, n_segments)
        #abs_pos: (batch, n_segments)
        #rel_pos: (batch, n_segments)
        batch, n_segments, _ = z_pn.shape
        _, max_track = prog.shape
        
        z_pn = z_pn + self.pe[:, :8*n_segments, :][:, ::8]
        z_pn = z_pn + self.total_len_embedding(total_len)
        z_pn = z_pn + self.abs_pos_embedding(abs_pos)
        z_pn = z_pn + self.rel_pos_embedding(rel_pos)
        
        z_pn = self.context_enc(z_pn) #(batch, n_segments, 256)
        z_pn = z_pn.unsqueeze(1) + self.prog_embedding(prog).unsqueeze(2)  #(batch, max_track, n_segments, 256)
        z_pn = z_pn.reshape(-1, n_segments, self.d_model)

        fn = fn.permute(0, 1, 3, 2).reshape(batch, -1, max_track)
        z_fn = self.func_embedding(fn)#(batch, 8*n_segments, max_track, d_model)
        
        z_fn = torch.cat([
                        self.start_embedding[prog].unsqueeze(1),   #(batch, 1, max_track, d_model)
                        z_fn[:, :-1]], 
                        dim=1) #batch, 8*n_segments, max_track, d_model

        z_fn = z_fn + self.prog_embedding(prog).unsqueeze(1) 

        z_fn = z_fn + self.pe[:, :z_fn.shape[1], :].unsqueeze(2)
        z_fn = z_fn + self.total_len_embedding(total_len).repeat_interleave(8, dim=1).unsqueeze(2)
        z_fn = z_fn + self.abs_pos_embedding(abs_pos).repeat_interleave(8, dim=1).unsqueeze(2)
        z_fn = z_fn + self.rel_pos_embedding(rel_pos).repeat_interleave(8, dim=1).unsqueeze(2)

        for layer in range(self.function_dec_layer):
            z_fn = z_fn.reshape(-1, max_track, self.d_model)
            z_fn = self.mt_trf[f'track_layer_{layer}'](src=z_fn, 
                                                    src_key_padding_mask=tk_mask.unsqueeze(1).repeat(1, 8*n_segments, 1).reshape(-1, max_track))
            z_fn = z_fn.reshape(batch, -1, max_track, self.d_model).permute(0, 2, 1, 3).reshape(-1, 8*n_segments, self.d_model)
            z_fn = self.mt_trf[f'time_layer_{layer}'](tgt=z_fn,
                                                    tgt_mask=self.generate_square_subsequent_mask(8*n_segments).to(z_fn.device),
                                                    tgt_key_padding_mask=tm_mask.unsqueeze(1).repeat(1, max_track, 1).reshape(-1, n_segments).repeat_interleave(8, dim=-1),
                                                    memory=z_pn)               
            z_fn = z_fn.reshape(batch, max_track, -1, self.d_model).permute(0, 2, 1, 3) #(batch, 8*n_segments, max_track, d_model)

        fn_logits = self.func_out_linear(z_fn)

        return fn_logits, fn

    def loss_function(self, fn_recon, fn_gt, tm_mask, tk_mask):

        mask = torch.logical_or(tm_mask.repeat_interleave(8, dim=-1).unsqueeze(-1), tk_mask.unsqueeze(1)) #(batch, 8*max_time, track)        
        unmask = torch.logical_not(mask)

        function_loss = self.criterion(fn_recon[unmask].reshape(-1, NUM_TIME_CODE), 
                            fn_gt[unmask].reshape(-1))
        return function_loss

    def loss(self, z_pn, prog, fn, tm_mask, tk_mask, total_len, abs_pos, rel_pos):
        output = self.run(z_pn, prog, fn, tm_mask, tk_mask, total_len, abs_pos, rel_pos)
        return self.loss_function(*output, tm_mask, tk_mask)

    def forward(self, mode, *input, **kwargs):
        if mode in ["run", 0]:
            return self.run(*input, **kwargs)
        elif mode in ['loss', 'train', 1]:
            return self.loss(*input, **kwargs)
        elif mode in ['inference', 'eval', 'val', 2]:
            return self.inference(*input, **kwargs)
        else:
            raise NotImplementedError
        
    def z_pn_encode(self, z_pn, prog, total_len, abs_pos, rel_pos):
        #encodes the first 16-segment chunk of z_pn
        z_pn_clip = z_pn[:, :16]
        clip_len = z_pn_clip.shape[1]
        z_pn_clip = z_pn_clip + self.pe[:, :8*clip_len, :][:, ::8]
        z_pn_clip = z_pn_clip + self.total_len_embedding(total_len[:, :16])
        z_pn_clip = z_pn_clip + self.abs_pos_embedding(abs_pos[:, :16])
        z_pn_clip = z_pn_clip + self.rel_pos_embedding(rel_pos[:, :16])

        z_pn_clip = self.context_enc(z_pn_clip) #(batch, num_bar, 256)
        z_pn_clip = z_pn_clip.unsqueeze(1) + self.prog_embedding(prog).unsqueeze(2)  #(batch, max_track, num_bar, 256)
        z_pn_clip = z_pn_clip.reshape(-1, clip_len, self.d_model)
        return z_pn_clip
    

    def inference(self, pno_red, prog, fn_prompt, total_len, abs_pos, rel_pos, blur=.5, p=.1, t=1):
        #pno_red: (batch, n_segments, 32, max_simu_note, 6) piano reduction
        #prog: (batch, max_track)
        #fn_prompt: (batch, 1, max_track, 32)   orchestral function prompt
        #total_len: (batch, n_segments)
        #abs_pos: (batch, n_segments)
        #rel_pos: (batch, n_segments)

        batch, n_segments, time, max_simu_note, _ = pno_red.shape
        _, max_track = prog.shape

        # encode content representation from piano reduction
        pno_red = pno_red.reshape(-1, time, max_simu_note, 6)
        z_pn_ = self.piano_encoder(pno_red)[0].mean.reshape(batch, n_segments, -1)    #(batch, n_segments, 256)
        z_pn = (1-blur)*z_pn_.clone() + blur*torch.empty(z_pn_.shape, device=z_pn_.device).normal_(mean=0, std=1) 
        z_pn_clip = self.z_pn_encode(z_pn, prog, total_len, abs_pos, rel_pos)
        
        # if orchestral function prompt provided, encode style representation from starting orchestral function
        if fn_prompt is not None:
            fn_prompt = fn_prompt.reshape(-1, 32)
            fn_prompt = self.function_encoder.get_code_indices(fn_prompt).reshape(batch, max_track, 8).permute(0, 2, 1)  #(batch, 8, max_track)

        #each track is initiated with a starting code based on its instrument
        start = self.start_embedding[prog].unsqueeze(1)  #(batch, 1, max_track, dmodel)
        
        # store the predicted orchestral function codes
        fn = torch.empty((batch, 0, max_track)).long().to(start.device)

        for idx in range(8*n_segments):
            if (idx < 8) and (fn_prompt is not None): #i.e., prompt provided
                start = torch.cat([start, self.func_embedding(fn[:, idx-1: idx, :])], dim=1)
                fn = torch.cat([fn, fn_prompt[:, idx: idx+1, :]], dim=1) 
                continue
            else:
                # get z_fn sequence chunk till the current step
                z_fn = torch.cat([start, self.func_embedding(fn[:, idx-1: idx, :])], dim=1)
                
                # forward 4 bars if predicted sequence exceeds 32 bars
                if z_fn.shape[1] > 128:
                    clip_start = ((z_fn.shape[1] - 129) // 16 + 1) * 16 + 1
                    z_fn = torch.cat([z_fn[:, 0: 1], z_fn[:, clip_start:]], dim=1)
                    if (z_fn.shape[1] % 16) == 1:
                        z_pn = z_pn[:, 2:]
                        total_len = total_len[:, 2:]
                        abs_pos = abs_pos[:, 2:]
                        rel_pos = rel_pos[:, 2:]
                        z_pn_clip = self.z_pn_encode(z_pn, prog, total_len, abs_pos, rel_pos)
                
                #print(z_fn.shape[1], total_len.shape[1])

                # add positional embeddings
                clip_len = z_fn.shape[1]
                z_fn = z_fn + self.prog_embedding(prog).unsqueeze(1)
                z_fn = z_fn + self.pe[:, :clip_len, :].unsqueeze(2)
                z_fn = z_fn + self.total_len_embedding(total_len[:, :16]).repeat_interleave(8, dim=1)[:, :clip_len].unsqueeze(2)
                z_fn = z_fn + self.abs_pos_embedding(abs_pos[:, :16]).repeat_interleave(8, dim=1)[:, :clip_len].unsqueeze(2)
                z_fn = z_fn + self.rel_pos_embedding(rel_pos[:, :16]).repeat_interleave(8, dim=1)[:, :clip_len].unsqueeze(2)

                #Transformer autoregressive prediction
                for layer in range(self.function_dec_layer):
                    z_fn = z_fn.reshape(-1, max_track, self.d_model)
                    z_fn = self.mt_trf[f'track_layer_{layer}'](src=z_fn)
                    z_fn = z_fn.reshape(batch, -1, max_track, self.d_model).permute(0, 2, 1, 3).reshape(-1, clip_len, self.d_model)
                    z_fn = self.mt_trf[f'time_layer_{layer}'](tgt=z_fn,
                                                            tgt_mask=self.generate_square_subsequent_mask(sz=clip_len).to(z_fn.device),
                                                            memory=z_pn_clip)               
                    z_fn = z_fn.reshape(batch, max_track, -1, self.d_model).permute(0, 2, 1, 3)
                
                # update function code chunk till the current step
                start = torch.cat([start, self.func_embedding(fn[:, idx-1: idx, :])], dim=1)

                # prediction for this step using temprature-controlled nucleus sampling
                fn_logits = self.func_out_linear(z_fn[:, -1,]) / t
                filtered_fn_logits = self.nucleus_filter(fn_logits, p)
                fn_probability = F.softmax(filtered_fn_logits, dim=-1)
                fn_pred = torch.multinomial(fn_probability.reshape(-1, NUM_TIME_CODE), 1).reshape(fn_probability.shape[:-1]).unsqueeze(1)
                # store the predicted orchestral function codes
                fn = torch.cat([fn, fn_pred], dim=1)
                if fn.shape[1] == 8*n_segments:
                    break
    
        fn = fn.reshape(batch, n_segments, 8, max_track).permute(0, 1, 3, 2)
        z_fn = self.function_encoder.infer_by_codes(fn)
        return self.autoencoder.infer_with_function_codes(z_pn_, prog.unsqueeze(1), z_fn)
    
    def nucleus_filter(self, logits, p):
        #sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        #cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        cum_sum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        #sorted_indices_to_remove = cumulative_probs > p
        nucleus = cum_sum_probs < p
        # Shift the indices to the right to keep also the first token above the threshold
        #sorted_indices_to_remove = torch.cat([sorted_indices_to_remove.new_zeros(sorted_indices_to_remove.shape[:-1] + (1,)), sorted_indices_to_remove[..., :-1]], dim=-1)
        nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
        nucleus = nucleus.gather(-1, sorted_indices.argsort(-1))

        logits[~nucleus] = float('-inf')
        return logits
    
    @classmethod
    def init_model(cls, pretrain_model_path=None, DEVICE='cuda:0'):
        """Fast model initialization."""
        autoencoder = Query_and_reArrange(name='pretrain', trf_layers=2, device=DEVICE)
        if pretrain_model_path is not None:
            autoencoder.load_state_dict(torch.load(pretrain_model_path, map_location=torch.device('cpu')))
        autoencoder.eval()
        model = cls(autoencoder.mixture_enc, autoencoder.function_enc, DEVICE=DEVICE).to(DEVICE)
        return model
    
    @classmethod
    def init_inference_model(cls, prior_model_path, autoencoder_path, DEVICE='cuda:0'):
        """Fast model initialization."""
        autoencoder = Query_and_reArrange(name='pretrain', trf_layers=2, device=DEVICE)
        autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=torch.device('cpu')))
        autoencoder.eval()
        model = cls(inference=True, autoencoder=autoencoder, DEVICE=DEVICE).to(DEVICE)
        model.load_state_dict(torch.load(prior_model_path), strict=False)
        return model
    

if __name__ == '__main__':
    from prior_dataset import VQ_LMD_Dataset, collate_fn
    import numpy as np
    import os
    os.environ['CUDA_VISIBLE_DEVICES']= '1'
    from torch.utils.data import DataLoader
    DEVICE = 'cuda:0'
    
    dataset = VQ_LMD_Dataset(lmd_dir="/data1/zhaojw/LMD/VQ-Q&A-T-009-reorder-random_discard/", debug_mode=True, split='Train', mode='train')
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda b: collate_fn(b, DEVICE))

    #model = Prior.init_model(pretrain_model_path="/data1/zhaojw/data_file_dir/params_autoencoder.pt", DEVICE=DEVICE)
    model = Prior.init_inference_model(prior_model_path="/data1/zhaojw/data_file_dir/params_prior.pt",\
                                       autoencoder_path="/data1/zhaojw/data_file_dir/params_autoencoder.pt",\
                                       DEVICE=DEVICE)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    for pno_red, programs, function, track_mask, time_mask, total_length, abs_pos, rel_pos in loader:
        loss = model('loss', pno_red, programs, function, track_mask, time_mask, total_length, abs_pos, rel_pos)
        print(loss)

        break