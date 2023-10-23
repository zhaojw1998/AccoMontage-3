import math
import torch
from torch import nn
import torch.nn.functional as F
from .QandA import QandA
from .TransformerEncoderLayer import TransformerEncoderLayer as TransformerEncoderLayerRPE
import numpy as np

NUM_INSTR_CLASS = 34
NUM_PITCH_CODE = 64
NUM_TIME_CODE = 128
TOTAL_LEN_BIN = np.array([4, 7, 12, 15, 20, 23, 28, 31, 36, 39, 44, 47, 52, 55, 60, 63, 68, 71, 76, 79, 84, 87, 92, 95, 100, 103, 108, 111, 116, 119, 124, 127, 132])
ABS_POS_BIN = np.arange(129)
REL_POS_BIN = np.arange(128)

class Prior(nn.Module):
    def __init__(self, mixture_encoder=None,
                       pitch_function_encoder=None,
                       time_function_encoder=None,
                       context_enc_layer=2, 
                       function_dec_layer=4, 
                       d_model=256, 
                       nhead=8, 
                       dim_feedforward=1024, 
                       dropout=.1, 
                       ft_resolution=8,
                       inference=False,
                       QaA_model=None,
                       DEVICE='cuda:0'):
        super(Prior, self).__init__()

        # embeddings
        self.fp_embedding = nn.Embedding(num_embeddings=NUM_PITCH_CODE+1, embedding_dim=d_model, padding_idx=NUM_PITCH_CODE)
        self.ft_embedding = nn.Embedding(num_embeddings=NUM_TIME_CODE+1, embedding_dim=d_model, padding_idx=NUM_TIME_CODE)
        self.prog_embedding = nn.Embedding(num_embeddings=NUM_INSTR_CLASS+1, embedding_dim=d_model, padding_idx=NUM_INSTR_CLASS)
        self.total_len_embedding = nn.Embedding(num_embeddings=len(TOTAL_LEN_BIN)+1, embedding_dim=d_model, padding_idx=len(TOTAL_LEN_BIN))
        self.abs_pos_embedding = nn.Embedding(num_embeddings=len(ABS_POS_BIN)+1, embedding_dim=d_model, padding_idx=len(ABS_POS_BIN))
        self.rel_pos_embedding = nn.Embedding(num_embeddings=len(REL_POS_BIN)+1, embedding_dim=d_model, padding_idx=len(REL_POS_BIN))
        
        self.start_embedding = nn.Parameter(torch.empty(NUM_INSTR_CLASS+1, 9, d_model))
        nn.init.normal_(self.start_embedding)
        with torch.no_grad():
                self.start_embedding[NUM_INSTR_CLASS].fill_(0)

        #pre-trained encoders
        if not inference:
            self.mixture_encoder = mixture_encoder
            for param in self.mixture_encoder.parameters():
                param.requires_grad = False
            self.pitch_function_encoder = pitch_function_encoder
            for param in self.pitch_function_encoder.parameters():
                param.requires_grad = False
            self.time_function_encoder = time_function_encoder
            for param in self.time_function_encoder.parameters():
                param.requires_grad = False
        else:
            self.QaA_model = QaA_model
            self.mixture_encoder = self.QaA_model.prmat_enc_fltn
            self.pitch_function_encoder = self.QaA_model.func_pitch_enc
            self.time_function_encoder = self.QaA_model.func_time_enc

        #multi-stream Transformer
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
        self.ms_trf = nn.ModuleDict({})
        for layer in range(function_dec_layer):
            """self.ms_trf[f'track_layer_{layer}'] = nn.TransformerEncoderLayer(d_model=d_model, 
                                                            nhead=nhead, 
                                                            dim_feedforward=dim_feedforward, 
                                                            dropout=dropout, 
                                                            activation=F.gelu, 
                                                            batch_first=True, 
                                                            norm_first=True,
                                                            device=DEVICE)"""
            self.ms_trf[f'track_layer_{layer}'] = TransformerEncoderLayerRPE(d_model=d_model, 
                                                            nhead=nhead, 
                                                            dim_feedforward=dim_feedforward, 
                                                            dropout=dropout, 
                                                            norm_first=True,
                                                            max_len=24).to(DEVICE)
            self.ms_trf[f'time_layer_{layer}'] = nn.TransformerDecoderLayer(d_model=d_model, 
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
        #pe = torch.flip(pe, dims=[1])
        pe = pe.to(DEVICE)
        self.register_buffer('pe', pe)
        
        #decoder output module 
        self.fp_out_linear = nn.Linear(d_model, NUM_PITCH_CODE)
        self.ft_out_linear = nn.Linear(d_model, NUM_TIME_CODE)

        #constants
        self.d_model = d_model
        self.function_dec_layer = function_dec_layer
        self.ft_resolution = ft_resolution

        #loss function
        self.criterion = nn.CrossEntropyLoss(reduction='mean')


    def generate_square_subsequent_mask(self, sz=15):
        return torch.triu(torch.ones(sz, sz), diagonal=1).repeat_interleave(9,dim=0).repeat_interleave(9,dim=1).bool()


    def run(self, mix, prog, fp, ft, tm_mask, tk_mask, total_len, abs_pos, rel_pos, inference=False):
        #mix: (batch, max_time, 256)
        #prog: (batch, max_track)
        #fp: (batch, max_time, max_track)
        #ft: (batch, max_time, max_track, 8)
        #tm_mask: (batch, max_time)
        #tk_mask: (batch, max_track)
        #total_len: (batch, max_time)
        #abs_pos: (batch, max_time)
        #rel_pos: (batch, max_time)
        batch, max_time, _ = mix.shape
        _, max_track = prog.shape

        #with torch.no_grad():
        #mix = mix.reshape(-1, time, max_simu_note, 6)
        #mix = self.mixture_encoder(mix)[0].mean.reshape(batch, num_2bar, -1)    #(batch, num_2bar, 256)
        #fp = fp.reshape(-1, 128)
        #fp = self.pitch_function_encoder.get_code_indices(fp).reshape(batch, num_2bar, max_track)
        #ft = ft.reshape(-1, 32)
        #ft = self.time_function_encoder.get_code_indices(ft).reshape(batch, num_2bar, max_track, self.ft_resolution)
        
        mix = mix + self.pe[:, :mix.shape[1], :]
        #mix = mix + self.total_len_embedding(total_len)
        #mix = mix + self.abs_pos_embedding(abs_pos)
        #mix = mix + self.rel_pos_embedding(rel_pos)
        mix = mix.unsqueeze(1) + self.prog_embedding(prog).unsqueeze(2)  #(batch, max_track, max_time, 256)
        mix = self.context_enc(mix.reshape(-1, max_time, self.d_model)) #(batch*max_track, max_time, 256)

        func = torch.cat([self.fp_embedding(fp[:, :-1].unsqueeze(-1)),
                        self.ft_embedding(ft[:, :-1])],
                        dim=-2) #batch, max_time-1, max_track, 9, d_model
        
        func = torch.cat([
                        self.start_embedding[prog].unsqueeze(1),   #(batch, 1, max_track, 9, d_model)
                        func], 
                        dim=1) #batch, max_time, max_track, 9, d_model
        
        func = func.permute(0, 1, 3, 2, 4).reshape(batch, -1, max_track, self.d_model) #(batch, max_time*9, max_track, d_model)

        func = func + self.prog_embedding(prog).unsqueeze(1) 
        func = func + self.pe[:, :func.shape[1], :].unsqueeze(2)
        func = func + self.total_len_embedding(total_len).repeat_interleave(9, dim=1).unsqueeze(2)
        func = func + self.abs_pos_embedding(abs_pos).repeat_interleave(9, dim=1).unsqueeze(2)
        func = func + self.rel_pos_embedding(rel_pos).repeat_interleave(9, dim=1).unsqueeze(2)

        for layer in range(self.function_dec_layer):
            func = func.reshape(-1, max_track, self.d_model)
            func = self.ms_trf[f'track_layer_{layer}'](src=func, 
                                                    src_key_padding_mask=tk_mask.unsqueeze(1).repeat(1, max_time*9, 1).reshape(-1, max_track))
            func = func.reshape(batch, -1, max_track, self.d_model).permute(0, 2, 1, 3).reshape(-1, max_time*9, self.d_model)
            func = self.ms_trf[f'time_layer_{layer}'](tgt=func,
                                                    tgt_mask=self.generate_square_subsequent_mask(max_time).to(func.device),
                                                    tgt_key_padding_mask=tm_mask.unsqueeze(1).repeat(1, max_track, 1).reshape(-1, max_time).repeat_interleave(9, dim=-1),
                                                    memory=mix)               
            func = func.reshape(batch, max_track, -1, self.d_model).permute(0, 2, 1, 3) #(batch, max_time*9, max_track, d_model)

        func = func.reshape(batch, max_time, 9, max_track, self.d_model)
        fp_recon = self.fp_out_linear(func[:, :, 0])
        ft_recon = self.ft_out_linear(func[:, :, 1:].permute(0, 1, 3, 2, 4))

        return fp_recon, ft_recon


    def loss_function(self, fp_recon, ft_recon, fp_gt, ft_gt, tm_mask, tk_mask):
        mask = torch.logical_or(tm_mask.unsqueeze(-1), tk_mask.unsqueeze(1))
        unmask = torch.logical_not(mask)

        fp_loss = self.criterion(fp_recon[unmask], 
                            fp_gt[unmask])
        ft_loss = self.criterion(ft_recon[unmask].reshape(-1, NUM_TIME_CODE), 
                            ft_gt[unmask].reshape(-1))
        
        loss = 0.11*fp_loss + 0.89*ft_loss
        return loss, fp_loss, ft_loss
    

    def loss(self, mix, prog, fp, ft, tm_mask, tk_mask, total_len, abs_pos, rel_pos):
        output = self.run(mix, prog, fp, ft, tm_mask, tk_mask, total_len, abs_pos, rel_pos, inference=False)
        return self.loss_function(*output, fp, ft, tm_mask, tk_mask)
    

    def forward(self, mode, *input, **kwargs):
        if mode in ["run", 0]:
            return self.run(*input, **kwargs)
        elif mode in ['loss', 'train', 1]:
            return self.loss(*input, **kwargs)
        elif mode in ['inference', 'eval', 'val', 2]:
            return self.inference(*input, **kwargs)
        else:
            raise NotImplementedError


    def run_autoregressive_greedy(self, mix, prog, fp, ft, total_len, abs_pos, rel_pos, blur=.5):
        #mix: (batch, num2bar, bar_resolution, max_simu_note, 6)
        #prog: (batch, max_track)
        #fp: (batch, 1, max_track, 128)
        #ft: (batch, 1, max_track, 32)
        #total_len: (batch, num2bar)
        #abs_pos: (batch, num2bar)
        #rel_pos: (batch, num2bar)
        batch, num_2bar, time, max_simu_note, _ = mix.shape
        _, max_track = prog.shape

        mix = mix.reshape(-1, time, max_simu_note, 6)
        mix = self.mixture_encoder(mix)[0].mean.reshape(batch, num_2bar, -1)    #(batch, num_2bar, 256)
        mix_ = (1-blur)*mix.clone() + blur*torch.empty(mix.shape, device=mix.device).normal_(mean=0, std=1) + self.pe[:, :mix.shape[1], :]
        #mix_ = mix_ + self.total_len_embedding(total_len)
        #mix_ = mix_ + self.abs_pos_embedding(abs_pos)
        #mix_ = mix_ + self.rel_pos_embedding(rel_pos)
        mix_ = mix_.unsqueeze(1) + self.prog_embedding(prog).unsqueeze(2)  #(batch, max_track, num2bar, 256)
        mix_ = self.context_enc(mix_.reshape(-1, num_2bar, self.d_model))
        
        func = self.start_embedding[prog].unsqueeze(1)   #(batch, 1, max_track, 9, d_model)
        for idx in range(num_2bar):
            if idx == 0:
                if (fp is not None) and (ft is not None):
                    fp = fp.reshape(-1, 128)
                    fp = self.pitch_function_encoder.get_code_indices(fp).reshape(batch, 1, max_track)
                    ft = ft.reshape(-1, 32)
                    ft = self.time_function_encoder.get_code_indices(ft).reshape(batch, 1, max_track, self.ft_resolution)
                    continue
                else:
                    fp = torch.empty((batch, 0, max_track)).long().to(mix.device)
                    ft = torch.empty((batch, 0, max_track, self.ft_resolution)).long().to(mix.device)                 
            elif idx > 0:
                func = torch.cat([
                                func, 
                                torch.cat([self.fp_embedding(fp[:, idx-1: idx].unsqueeze(-1)),
                                            self.ft_embedding(ft[:, idx-1: idx])],
                                            dim=-2) #*batch, 1, max_track, 9, d_model
                                ], dim=1) #*batch, idx+1, max_track, 9, d_model
            
            func = func.permute(0, 1, 3, 2, 4).reshape(batch, -1, max_track, self.d_model)

            func = func + self.prog_embedding(prog).unsqueeze(1)
            func = func + self.pe[:, :func.shape[1], :].unsqueeze(2)
            func = func + self.total_len_embedding(total_len[:, : 1+idx]).repeat_interleave(9, dim=1).unsqueeze(2)
            func = func + self.abs_pos_embedding(abs_pos[:, : 1+idx]).repeat_interleave(9, dim=1).unsqueeze(2)
            func = func + self.rel_pos_embedding(rel_pos[:, : 1+idx]).repeat_interleave(9, dim=1).unsqueeze(2)

            for layer in range(self.function_dec_layer):
                  
                func = func.reshape(-1, max_track, self.d_model)
                func = self.ms_trf[f'track_layer_{layer}'](src=func)
                func = func.reshape(batch, -1, max_track, self.d_model).permute(0, 2, 1, 3).reshape(-1, (1+idx)*9, self.d_model)
                func = self.ms_trf[f'time_layer_{layer}'](tgt=func,
                                                        tgt_mask=self.generate_square_subsequent_mask(sz=1+idx).to(func.device),
                                                        memory=mix_)               
                func = func.reshape(batch, max_track, -1, self.d_model).permute(0, 2, 1, 3) #(batch, num2bar-1, max_track, d_model)
                #print('func output', func.shape)

            func = func.reshape(batch, 1+idx, 9, max_track, self.d_model).permute(0, 1, 3, 2, 4)
            fp_pred = self.fp_out_linear(func[:, -1, :, 0]).unsqueeze(1).max(-1)[1]
            ft_pred = self.ft_out_linear(func[:, -1, :, 1:]).unsqueeze(1).max(-1)[1]

            fp = torch.cat([fp, fp_pred], dim=1)
            ft = torch.cat([ft, ft_pred], dim=1)
            if fp.shape[1] == num_2bar:
                break
        
        z_fp = self.pitch_function_encoder.infer_by_codes(fp)
        z_ft = self.time_function_encoder.infer_by_codes(ft)
        return self.QaA_model.infer_with_function_codes(mix[0], prog[0].repeat(num_2bar, 1), z_fp[0], z_ft[0])
    

    def run_autoregressive_nucleus(self, mix, prog, fp, ft, total_len, abs_pos, rel_pos, blur=.5, p=.1, t=1):
        #mix: (batch, num2bar, bar_resolution, max_simu_note, 6)
        #prog: (batch, max_track)
        #fp: (batch, 1, max_track, 128)
        #ft: (batch, 1, max_track, 32)
        #total_len: (batch, num2bar)
        #abs_pos: (batch, num2bar)
        #rel_pos: (batch, num2bar)
        batch, num_2bar, time, max_simu_note, _ = mix.shape
        _, max_track = prog.shape

        mix = mix.reshape(-1, time, max_simu_note, 6)
        mix = self.mixture_encoder(mix)[0].mean.reshape(batch, num_2bar, -1)    #(batch, num_2bar, 256)
        mix_ = (1-blur)*mix.clone() + blur*torch.empty(mix.shape, device=mix.device).normal_(mean=0, std=1) + self.pe[:, :mix.shape[1], :]
        #mix_ = mix_ + self.total_len_embedding(total_len)
        #mix_ = mix_ + self.abs_pos_embedding(abs_pos)
        #mix_ = mix_ + self.rel_pos_embedding(rel_pos)
        mix_ = mix_.unsqueeze(1) + self.prog_embedding(prog).unsqueeze(2)  #(batch, max_track, num2bar, 256)
        mix_ = self.context_enc(mix_.reshape(-1, num_2bar, self.d_model))
        
        func = self.start_embedding[prog].unsqueeze(1)   #(batch, 1, max_track, 9, d_model)
        for idx in range(num_2bar):
            if idx == 0:
                if (fp is not None) and (ft is not None):
                    fp = fp.reshape(-1, 128)
                    fp = self.pitch_function_encoder.get_code_indices(fp).reshape(batch, 1, max_track)
                    ft = ft.reshape(-1, 32)
                    ft = self.time_function_encoder.get_code_indices(ft).reshape(batch, 1, max_track, self.ft_resolution)
                    continue
                else:
                    fp = torch.empty((batch, 0, max_track)).long().to(mix.device)
                    ft = torch.empty((batch, 0, max_track, self.ft_resolution)).long().to(mix.device)                 
            elif idx > 0:
                func = torch.cat([
                                func, 
                                torch.cat([self.fp_embedding(fp[:, idx-1: idx].unsqueeze(-1)),
                                            self.ft_embedding(ft[:, idx-1: idx])],
                                            dim=-2) #*batch, 1, max_track, 9, d_model
                                ], dim=1) #*batch, idx+1, max_track, 9, d_model
                
            func = func.permute(0, 1, 3, 2, 4).reshape(batch, -1, max_track, self.d_model)

            func = func + self.prog_embedding(prog).unsqueeze(1)
            func = func + self.pe[:, :func.shape[1], :].unsqueeze(2)
            func = func + self.total_len_embedding(total_len[:, : 1+idx]).repeat_interleave(9, dim=1).unsqueeze(2)
            func = func + self.abs_pos_embedding(abs_pos[:, : 1+idx]).repeat_interleave(9, dim=1).unsqueeze(2)
            func = func + self.rel_pos_embedding(rel_pos[:, : 1+idx]).repeat_interleave(9, dim=1).unsqueeze(2)

            for layer in range(self.function_dec_layer):
                  
                func = func.reshape(-1, max_track, self.d_model)
                func = self.ms_trf[f'track_layer_{layer}'](src=func)
                func = func.reshape(batch, -1, max_track, self.d_model).permute(0, 2, 1, 3).reshape(-1, (1+idx)*9, self.d_model)
                func = self.ms_trf[f'time_layer_{layer}'](tgt=func,
                                                        tgt_mask=self.generate_square_subsequent_mask(sz=1+idx).to(func.device),
                                                        memory=mix_)               
                func = func.reshape(batch, max_track, -1, self.d_model).permute(0, 2, 1, 3) #(batch, num2bar-1, max_track, d_model)
                #print('func output', func.shape)

            func = func.reshape(batch, 1+idx, 9, max_track, self.d_model).permute(0, 1, 3, 2, 4)
            
            
            fp_logits = self.fp_out_linear(func[:, -1, :, 0]).unsqueeze(1) / t
            if idx == 0:
                filtered_fp_logits = self.nucleus_filter(fp_logits/2, 2*p)
            else:
                filtered_fp_logits = self.nucleus_filter(fp_logits, p)
            fp_probability = F.softmax(filtered_fp_logits, dim=-1)
            #print('fp_probability', fp_probability.shape)
            fp_pred = torch.multinomial(fp_probability.reshape(-1, NUM_PITCH_CODE), 1).reshape(fp_probability.shape[:-1])

            ft_logits = self.ft_out_linear(func[:, -1, :, 1:]).unsqueeze(1) / t
            if idx == 0:
                filtered_ft_logits = self.nucleus_filter(ft_logits/2, 2*p)
            else:
                filtered_ft_logits = self.nucleus_filter(ft_logits, p)
            ft_probability = F.softmax(filtered_ft_logits, dim=-1)
            ft_pred = torch.multinomial(ft_probability.reshape(-1, NUM_TIME_CODE), 1).reshape(ft_probability.shape[:-1])

            fp = torch.cat([fp, fp_pred], dim=1)
            ft = torch.cat([ft, ft_pred], dim=1)
            if fp.shape[1] == num_2bar:
                break
        
        z_fp = self.pitch_function_encoder.infer_by_codes(fp)
        z_ft = self.time_function_encoder.infer_by_codes(ft)
        return self.QaA_model.infer_with_function_codes(mix[0], prog[0].repeat(num_2bar, 1), z_fp[0], z_ft[0])
    
    def nucleus_filter(self, logits, p):
        #sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        #cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        cum_sum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        nucleus = cum_sum_probs < p
        # Shift the indices to the right to keep also the first token above the threshold
        nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
        nucleus = nucleus.gather(-1, sorted_indices.argsort(-1))
        logits[~nucleus] = float('-inf')
        return logits
    

    def run_autoregressive_nucleus_long_sample(self, mix, prog, fp, ft, total_len, abs_pos, rel_pos, blur=.5, p=.1, t=1):
        #mix: (batch, num2bar, bar_resolution, max_simu_note, 6)
        #prog: (batch, max_track)
        #fp: (batch, 1, max_track, 128)
        #ft: (batch, 1, max_track, 32)
        #total_len: (batch, num2bar)
        #abs_pos: (batch, num2bar)
        #rel_pos: (batch, num2bar)
        batch, num_2bar, time, max_simu_note, _ = mix.shape
        _, max_track = prog.shape

        MAX_LEN = 16
        HOP_LEN = 4
        START = 0

        mix = mix.reshape(-1, time, max_simu_note, 6)
        mix = self.mixture_encoder(mix)[0].mean.reshape(batch, num_2bar, -1)    #(batch, num_2bar, 256)

        func = self.start_embedding[prog].unsqueeze(1)   #(batch, 1, max_track, 9, d_model)
        for START in range(0, num_2bar - MAX_LEN+1, HOP_LEN):
            mix_ = (1-blur)*mix[:, START: START+MAX_LEN].clone() 
            mix_ = mix_ + blur*torch.empty(mix_.shape, device=mix_.device).normal_(mean=0, std=1) + self.pe[:, :MAX_LEN, :]
            #mix_ = mix_ + self.total_len_embedding(total_len)
            #mix_ = mix_ + self.abs_pos_embedding(abs_pos)
            #mix_ = mix_ + self.rel_pos_embedding(rel_pos)
            mix_ = mix_.unsqueeze(1) + self.prog_embedding(prog).unsqueeze(2)  #(batch, max_track, num2bar, 256)
            mix_ = self.context_enc(mix_.reshape(-1, MAX_LEN, self.d_model))

            if START == 0:
                init = 0
            else:
                init = MAX_LEN-HOP_LEN
            for idx in range(init, MAX_LEN):
                if START == 0:
                    if idx == 0:
                        if (fp is not None) and (ft is not None):
                            fp = fp.reshape(-1, 128)
                            fp = self.pitch_function_encoder.get_code_indices(fp).reshape(batch, 1, max_track)
                            ft = ft.reshape(-1, 32)
                            ft = self.time_function_encoder.get_code_indices(ft).reshape(batch, 1, max_track, self.ft_resolution)
                            continue
                        else:
                            fp = torch.empty((batch, 0, max_track)).long().to(mix.device)
                            ft = torch.empty((batch, 0, max_track, self.ft_resolution)).long().to(mix.device)                 
                    elif idx > 0:
                        func = torch.cat([
                                        func, 
                                        torch.cat([self.fp_embedding(fp[:, idx-1: idx].unsqueeze(-1)),
                                                    self.ft_embedding(ft[:, idx-1: idx])],
                                                    dim=-2) #*batch, 1, max_track, 9, d_model
                                        ], dim=1) #*batch, idx+1, max_track, 9, d_model
                else:
                    func = torch.cat([
                                        func, 
                                        torch.cat([self.fp_embedding(fp[:, idx-1: idx].unsqueeze(-1)),
                                                    self.ft_embedding(ft[:, idx-1: idx])],
                                                    dim=-2) #*batch, 1, max_track, 9, d_model
                                        ], dim=1) #*batch, idx+1, max_track, 9, d_model
                    if idx == init:
                        func = func[:, HOP_LEN:]

    
                func = func.permute(0, 1, 3, 2, 4).reshape(batch, -1, max_track, self.d_model)

                func = func + self.prog_embedding(prog).unsqueeze(1)
                func = func + self.pe[:, 9:9+func.shape[1], :].unsqueeze(2)
                print(START)
                print('func', func.shape), print('total_len', self.total_len_embedding(total_len[:, START: START+1+idx]).repeat_interleave(9, dim=1).unsqueeze(2).shape)
                func = func + self.total_len_embedding(total_len[:, START: START+1+idx]).repeat_interleave(9, dim=1).unsqueeze(2)
                func = func + self.abs_pos_embedding(abs_pos[:, START: START+1+idx]).repeat_interleave(9, dim=1).unsqueeze(2)
                func = func + self.rel_pos_embedding(rel_pos[:, START: START+1+idx]).repeat_interleave(9, dim=1).unsqueeze(2)

                for layer in range(self.function_dec_layer):
                  
                    func = func.reshape(-1, max_track, self.d_model)
                    func = self.ms_trf[f'track_layer_{layer}'](src=func)
                    func = func.reshape(batch, -1, max_track, self.d_model).permute(0, 2, 1, 3).reshape(-1, (1+idx)*9, self.d_model)
                    func = self.ms_trf[f'time_layer_{layer}'](tgt=func,
                                                            tgt_mask=self.generate_square_subsequent_mask(sz=1+idx).to(func.device),
                                                            memory=mix_)               
                    func = func.reshape(batch, max_track, -1, self.d_model).permute(0, 2, 1, 3) #(batch, num2bar-1, max_track, d_model)
                    #print('func output', func.shape)
                func = func.reshape(batch, 1+idx, 9, max_track, self.d_model).permute(0, 1, 3, 2, 4)
                
                fp_logits = self.fp_out_linear(func[:, -1, :, 0]).unsqueeze(1) / t
                if idx == 0:
                    filtered_fp_logits = self.nucleus_filter(fp_logits/2, 2*p)
                else:
                    filtered_fp_logits = self.nucleus_filter(fp_logits, p)
                fp_probability = F.softmax(filtered_fp_logits, dim=-1)
                #print('fp_probability', fp_probability.shape)
                fp_pred = torch.multinomial(fp_probability.reshape(-1, NUM_PITCH_CODE), 1).reshape(fp_probability.shape[:-1])

                ft_logits = self.ft_out_linear(func[:, -1, :, 1:]).unsqueeze(1) / t
                if idx == 0:
                    filtered_ft_logits = self.nucleus_filter(ft_logits/2, 2*p)
                else:
                    filtered_ft_logits = self.nucleus_filter(ft_logits, p)
                ft_probability = F.softmax(filtered_ft_logits, dim=-1)
                ft_pred = torch.multinomial(ft_probability.reshape(-1, NUM_TIME_CODE), 1).reshape(ft_probability.shape[:-1])

                fp = torch.cat([fp, fp_pred], dim=1)
                ft = torch.cat([ft, ft_pred], dim=1)
                if fp.shape[1] == num_2bar:
                    print('precise')
                    break

        
        if START + MAX_LEN < num_2bar:
            rest = num_2bar - (START + MAX_LEN)
            START = num_2bar - MAX_LEN
            mix_ = (1-blur)*mix[:, START:].clone() 
            mix_ = mix_ + blur*torch.empty(mix_.shape, device=mix_.device).normal_(mean=0, std=1) + self.pe[:, :MAX_LEN, :]
            mix_ = mix_.unsqueeze(1) + self.prog_embedding(prog).unsqueeze(2)  #(batch, max_track, num2bar, 256)
            mix_ = self.context_enc(mix_.reshape(-1, MAX_LEN, self.d_model))

            for idx in range(MAX_LEN - rest, MAX_LEN):
                func = torch.cat([
                                func, 
                                torch.cat([self.fp_embedding(fp[:, idx-1: idx].unsqueeze(-1)),
                                            self.ft_embedding(ft[:, idx-1: idx])],
                                            dim=-2) #*batch, 1, max_track, 9, d_model
                                ], dim=1) #*batch, idx+1, max_track, 9, d_model
                if idx == MAX_LEN - rest:
                    func = func[:, rest:]

                func = func.permute(0, 1, 3, 2, 4).reshape(batch, -1, max_track, self.d_model)
                
                func = func + self.prog_embedding(prog).unsqueeze(1)
                func = func + self.pe[:, 9:9+func.shape[1], :].unsqueeze(2)
                print(START)
                print('func', func.shape), print('total_len', self.total_len_embedding(total_len[:, START: START+1+idx]).repeat_interleave(9, dim=1).unsqueeze(2).shape)
                func = func + self.total_len_embedding(total_len[:, START: START+1+idx]).repeat_interleave(9, dim=1).unsqueeze(2)
                func = func + self.abs_pos_embedding(abs_pos[:, START: START+1+idx]).repeat_interleave(9, dim=1).unsqueeze(2)
                func = func + self.rel_pos_embedding(rel_pos[:, START: START+1+idx]).repeat_interleave(9, dim=1).unsqueeze(2)

                for layer in range(self.function_dec_layer):
                  
                    func = func.reshape(-1, max_track, self.d_model)
                    func = self.ms_trf[f'track_layer_{layer}'](src=func)
                    func = func.reshape(batch, -1, max_track, self.d_model).permute(0, 2, 1, 3).reshape(-1, (1+idx)*9, self.d_model)
                    func = self.ms_trf[f'time_layer_{layer}'](tgt=func,
                                                            tgt_mask=self.generate_square_subsequent_mask(sz=1+idx).to(func.device),
                                                            memory=mix_)               
                    func = func.reshape(batch, max_track, -1, self.d_model).permute(0, 2, 1, 3) #(batch, num2bar-1, max_track, d_model)
                    #print('func output', func.shape)
                func = func.reshape(batch, 1+idx, 9, max_track, self.d_model).permute(0, 1, 3, 2, 4)
                
                fp_logits = self.fp_out_linear(func[:, -1, :, 0]).unsqueeze(1) / t
                if idx == 0:
                    filtered_fp_logits = self.nucleus_filter(fp_logits/2, 2*p)
                else:
                    filtered_fp_logits = self.nucleus_filter(fp_logits, p)
                fp_probability = F.softmax(filtered_fp_logits, dim=-1)
                #print('fp_probability', fp_probability.shape)
                fp_pred = torch.multinomial(fp_probability.reshape(-1, NUM_PITCH_CODE), 1).reshape(fp_probability.shape[:-1])

                ft_logits = self.ft_out_linear(func[:, -1, :, 1:]).unsqueeze(1) / t
                if idx == 0:
                    filtered_ft_logits = self.nucleus_filter(ft_logits/2, 2*p)
                else:
                    filtered_ft_logits = self.nucleus_filter(ft_logits, p)
                ft_probability = F.softmax(filtered_ft_logits, dim=-1)
                ft_pred = torch.multinomial(ft_probability.reshape(-1, NUM_TIME_CODE), 1).reshape(ft_probability.shape[:-1])

                fp = torch.cat([fp, fp_pred], dim=1)
                ft = torch.cat([ft, ft_pred], dim=1)
                if fp.shape[1] == num_2bar:
                    break
        
        z_fp = self.pitch_function_encoder.infer_by_codes(fp)
        z_ft = self.time_function_encoder.infer_by_codes(ft)
        return self.QaA_model.infer_with_function_codes(mix[0], prog[0].repeat(num_2bar, 1), z_fp[0], z_ft[0])


    @classmethod
    def init_model(cls, pretrain_model_path=None, DEVICE='cuda:0'):
        """Fast model initialization."""
        vqQaA = QandA(name='pretrain', trf_layers=2, device=DEVICE)
        if pretrain_model_path is not None:
            vqQaA.load_state_dict(torch.load(pretrain_model_path, map_location=torch.device('cpu')))
        vqQaA.eval()
        model = cls(vqQaA.prmat_enc_fltn, vqQaA.func_pitch_enc, vqQaA.func_time_enc, DEVICE=DEVICE).to(DEVICE)
        return model
    
    @classmethod
    def init_inference_model(cls, prior_model_path, QaA_model_path, DEVICE='cuda:0'):
        """Fast model initialization."""
        vqQaA = QandA(name='pretrain', trf_layers=2, device=DEVICE)
        vqQaA.load_state_dict(torch.load(QaA_model_path, map_location=torch.device('cpu')))
        vqQaA.eval()
        model = cls(inference=True, QaA_model=vqQaA, DEVICE=DEVICE).to(DEVICE)
        model.load_state_dict(torch.load(prior_model_path), strict=False)
        return model
    
