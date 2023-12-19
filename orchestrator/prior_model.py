import math
import random
import torch
from torch import nn
import torch.nn.functional as F
from .QA_model import Query_and_reArrange
from .TransformerEncoderLayer import TransformerEncoderLayer as TransformerEncoderLayerRPE
from .prior_dataset import NUM_INSTR_CLASS, NUM_TIME_CODE, TOTAL_LEN_BIN, ABS_POS_BIN, REL_POS_BIN

class Prior(nn.Module):
    def __init__(self, mixture_encoder=None,
                       function_encoder=None,
                       context_enc_layer=12, 
                       function_dec_layer=12, 
                       d_model=256, 
                       nhead=8, 
                       dim_feedforward=1024, 
                       dropout=.1, 
                       function_resolution=8,
                       inference=False,
                       QA_model=None,
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
            self.mixture_encoder = mixture_encoder
            for param in self.mixture_encoder.parameters():
                param.requires_grad = False
            self.function_encoder = function_encoder
            for param in self.function_encoder.parameters():
                param.requires_grad = False
        else:
            self.QA_model = QA_model
            self.mixture_encoder = self.QA_model.mixture_enc
            self.function_encoder = self.QA_model.function_enc

        
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
        self.func_res = function_resolution

        #loss function
        self.criterion = nn.CrossEntropyLoss(reduction='mean')


    def generate_square_subsequent_mask(self, sz=15):
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool()


    def func_get_next_token(self, token, gt=None):
        #token: (batch, codebook_size)
        #gt: (bs,)
        if gt is None:
            idx = token.max(-1)[1]
        else:
            idx = gt
        token = torch.zeros_like(token, device=token.device)
        arange = torch.arange(token.shape[0], device=token.device).long()
        token[arange, idx] = 1
        return token.unsqueeze(1)   #one-hot shaoe (batch, 1, ft_codebook_size)

        


    def run(self, mix, prog, function, tm_mask, tk_mask, total_len, abs_pos, rel_pos, inference=False):
        #mix: (batch, max_time, 256)
        #prog: (batch, max_track)
        #function: (batch, max_time, max_track, 8)
        #tm_mask: (batch, max_time)
        #tk_mask: (batch, max_track)
        #total_len: (batch, max_time)
        #abs_pos: (batch, max_time)
        #rel_pos: (batch, max_time)
        batch, max_time, _ = mix.shape
        _, max_track = prog.shape
        
        mix = mix + self.pe[:, :self.func_res*mix.shape[1], :][:, ::self.func_res]
        mix = mix + self.total_len_embedding(total_len)
        mix = mix + self.abs_pos_embedding(abs_pos)
        mix = mix + self.rel_pos_embedding(rel_pos)
        
        mix = self.context_enc(mix) #(batch, max_time, 256)
        mix = mix.unsqueeze(1) + self.prog_embedding(prog).unsqueeze(2)  #(batch, max_track, max_time, 256)
        mix = mix.reshape(-1, max_time, self.d_model)

        function = function.permute(0, 1, 3, 2).reshape(batch, -1, max_track)
        func = self.func_embedding(function)#(batch, 8*max_time, max_track, d_model)
        
        func = torch.cat([
                        self.start_embedding[prog].unsqueeze(1),   #(batch, 1, max_track, d_model)
                        func[:, :-1]], 
                        dim=1) #batch, 8*max_time, max_track, d_model

        func = func + self.prog_embedding(prog).unsqueeze(1) 

        func = func + self.pe[:, :func.shape[1], :].unsqueeze(2)
        func = func + self.total_len_embedding(total_len).repeat_interleave(self.func_res, dim=1).unsqueeze(2)
        func = func + self.abs_pos_embedding(abs_pos).repeat_interleave(self.func_res, dim=1).unsqueeze(2)
        func = func + self.rel_pos_embedding(rel_pos).repeat_interleave(self.func_res, dim=1).unsqueeze(2)

        for layer in range(self.function_dec_layer):
            func = func.reshape(-1, max_track, self.d_model)
            func = self.mt_trf[f'track_layer_{layer}'](src=func, 
                                                    src_key_padding_mask=tk_mask.unsqueeze(1).repeat(1, self.func_res*max_time, 1).reshape(-1, max_track))
            func = func.reshape(batch, -1, max_track, self.d_model).permute(0, 2, 1, 3).reshape(-1, self.func_res*max_time, self.d_model)
            func = self.mt_trf[f'time_layer_{layer}'](tgt=func,
                                                    tgt_mask=self.generate_square_subsequent_mask(self.func_res*max_time).to(func.device),
                                                    tgt_key_padding_mask=tm_mask.unsqueeze(1).repeat(1, max_track, 1).reshape(-1, max_time).repeat_interleave(self.func_res, dim=-1),
                                                    memory=mix)               
            func = func.reshape(batch, max_track, -1, self.d_model).permute(0, 2, 1, 3) #(batch, 8*max_time, max_track, d_model)

        function_recon = self.func_out_linear(func)

        return function_recon, function

    

    def loss_function(self, function_recon, function_gt, tm_mask, tk_mask):

        mask = torch.logical_or(tm_mask.repeat_interleave(8, dim=-1).unsqueeze(-1), tk_mask.unsqueeze(1)) #(batch, 8*max_time, track)        
        unmask = torch.logical_not(mask)

        function_loss = self.criterion(function_recon[unmask].reshape(-1, NUM_TIME_CODE), 
                            function_gt[unmask].reshape(-1))
        return function_loss
    

    def loss(self, mix, prog, function, tm_mask, tk_mask, total_len, abs_pos, rel_pos):
        output = self.run(mix, prog, function, tm_mask, tk_mask, total_len, abs_pos, rel_pos, inference=False)
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


    def run_autoregressive_greedy(self, mix, prog, function, total_len, abs_pos, rel_pos, blur=.5):
        #mix: (batch, num2bar, bar_resolution, max_simu_note, 6)
        #prog: (batch, max_track)
        #function: (batch, 1, max_track, 32)
        #total_len: (batch, num2bar)
        #abs_pos: (batch, num2bar)
        #rel_pos: (batch, num2bar)
        batch, num_2bar, time, max_simu_note, _ = mix.shape
        _, max_track = prog.shape

        mix = mix.reshape(-1, time, max_simu_note, 6)
        mix = self.mixture_encoder(mix)[0].mean.reshape(batch, num_2bar, -1)    #(batch, num_2bar, 256)
        mix_ = (1-blur)*mix.clone() + blur*torch.empty(mix.shape, device=mix.device).normal_(mean=0, std=1) 
        
        mix_ = mix_ + self.pe[:, :self.func_res*mix.shape[1], :][:, ::self.func_res]
        mix_ = mix_ + self.total_len_embedding(total_len)
        mix_ = mix_ + self.abs_pos_embedding(abs_pos)
        mix_ = mix_ + self.rel_pos_embedding(rel_pos)

        mix_ = self.context_enc(mix_) #(batch, num_bar, 256)
        mix_ = mix_.unsqueeze(1) + self.prog_embedding(prog).unsqueeze(2)  #(batch, max_track, num_bar, 256)
        mix_ = mix_.reshape(-1, num_2bar, self.d_model)
        
        function = function.reshape(-1, 32)
        function = self.function_encoder.get_code_indices(function).reshape(batch, max_track, self.func_res)


        for idx in range(self.func_res, self.func_res*num_2bar):
            func = self.func_embedding(function) #*batch, max_track, 8, d_model
            func = func.permute(0, 2, 1, 3).reshape(batch, -1, max_track, self.d_model)

            func = func + self.prog_embedding(prog).unsqueeze(1)
            func = func + self.pe[:, :func.shape[1], :].unsqueeze(2)

            func = func + self.total_len_embedding(total_len).repeat_interleave(self.func_res, dim=1)[:, :func.shape[1]].unsqueeze(2)
            func = func + self.abs_pos_embedding(abs_pos).repeat_interleave(self.func_res, dim=1)[:, :func.shape[1]].unsqueeze(2)
            func = func + self.rel_pos_embedding(rel_pos).repeat_interleave(self.func_res, dim=1)[:, :func.shape[1]].unsqueeze(2)

            for layer in range(self.function_dec_layer):
                  
                func = func.reshape(-1, max_track, self.d_model)
                func = self.mt_trf[f'track_layer_{layer}'](src=func)
                func = func.reshape(batch, -1, max_track, self.d_model).permute(0, 2, 1, 3).reshape(-1, idx, self.d_model)
                func = self.mt_trf[f'time_layer_{layer}'](tgt=func,
                                                        tgt_mask=self.generate_square_subsequent_mask(sz=idx).to(func.device),
                                                        memory=mix_)               
                func = func.reshape(batch, max_track, -1, self.d_model).permute(0, 2, 1, 3) #(batch, num2bar-1, max_track, d_model)

            
            func_pred = self.func_out_linear(func[:, -1,]).max(-1)[1].unsqueeze(-1)

            function = torch.cat([function, func_pred], dim=-1)
            if function.shape[1] == self.func_res*num_2bar:
                break
        
        function = function.reshape(batch, max_track, num_2bar, self.func_res).permute(0, 2, 1, 3)
        z_func = self.function_encoder.infer_by_codes(function)
        return self.QA_model.infer_with_function_codes(mix[0], prog[0].repeat(num_2bar, 1), z_func[0])
    

    def run_autoregressive_nucleus(self, mix, prog, func_prompt, total_len, abs_pos, rel_pos, blur=.5, p=.1, t=1):
        #mix: (batch, num2bar, bar_resolution, max_simu_note, 6)
        #prog: (batch, max_track)
        #func_prompt: (batch, 1, max_track, 32)
        #total_len: (batch, num2bar)
        #abs_pos: (batch, num2bar)
        #rel_pos: (batch, num2bar)

        batch, num_2bar, time, max_simu_note, _ = mix.shape
        _, max_track = prog.shape

        mix = mix.reshape(-1, time, max_simu_note, 6)
        mix = self.mixture_encoder(mix)[0].mean.reshape(batch, num_2bar, -1)    #(batch, num_2bar, 256)
        mix_ = (1-blur)*mix.clone() + blur*torch.empty(mix.shape, device=mix.device).normal_(mean=0, std=1) 
        
        mix_ = mix_ + self.pe[:, :self.func_res*mix.shape[1], :][:, ::self.func_res]
        mix_ = mix_ + self.total_len_embedding(total_len)
        mix_ = mix_ + self.abs_pos_embedding(abs_pos)
        mix_ = mix_ + self.rel_pos_embedding(rel_pos)

        mix_ = self.context_enc(mix_) #(batch, num_bar, 256)
        mix_ = mix_.unsqueeze(1) + self.prog_embedding(prog).unsqueeze(2)  #(batch, max_track, num_bar, 256)
        mix_ = mix_.reshape(-1, num_2bar, self.d_model)
        
        start = self.start_embedding[prog].unsqueeze(1)  #(batch, 1, max_track, dmodel)

        if func_prompt is not None:
            func_prompt = func_prompt.reshape(-1, 32)
            func_prompt = self.function_encoder.get_code_indices(func_prompt).reshape(batch, max_track, self.func_res).permute(0, 2, 1)  #(batch, 8, max_track)
        #else:
        function = torch.empty((batch, 0, max_track)).long().to(mix.device)

        for idx in range(self.func_res*num_2bar):
            if (idx < self.func_res) and (func_prompt is not None):
                start = torch.cat([start, self.func_embedding(function[:, idx-1: idx, :])], dim=1)
                function = torch.cat([function, func_prompt[:, idx: idx+1, :]], dim=1) 
                continue
            else:
                func = torch.cat([start, self.func_embedding(function[:, idx-1: idx, :])], dim=1)

                func = func + self.prog_embedding(prog).unsqueeze(1)
                func = func + self.pe[:, :func.shape[1], :].unsqueeze(2)

                func = func + self.total_len_embedding(total_len).repeat_interleave(self.func_res, dim=1)[:, :func.shape[1]].unsqueeze(2)
                func = func + self.abs_pos_embedding(abs_pos).repeat_interleave(self.func_res, dim=1)[:, :func.shape[1]].unsqueeze(2)
                func = func + self.rel_pos_embedding(rel_pos).repeat_interleave(self.func_res, dim=1)[:, :func.shape[1]].unsqueeze(2)

                for layer in range(self.function_dec_layer):
                    
                    func = func.reshape(-1, max_track, self.d_model)
                    func = self.mt_trf[f'track_layer_{layer}'](src=func)
                    func = func.reshape(batch, -1, max_track, self.d_model).permute(0, 2, 1, 3).reshape(-1, idx+1, self.d_model)
                    func = self.mt_trf[f'time_layer_{layer}'](tgt=func,
                                                            tgt_mask=self.generate_square_subsequent_mask(sz=idx+1).to(func.device),
                                                            memory=mix_)               
                    func = func.reshape(batch, max_track, -1, self.d_model).permute(0, 2, 1, 3)#(batch, num2bar-1, max_track, d_model)
                
                start = torch.cat([start, self.func_embedding(function[:, idx-1: idx, :])], dim=1)

                func_logits = self.func_out_linear(func[:, -1,]) / t
                filtered_func_logits = self.nucleus_filter(func_logits, p)
                func_probability = F.softmax(filtered_func_logits, dim=-1)
                func_pred = torch.multinomial(func_probability.reshape(-1, NUM_TIME_CODE), 1).reshape(func_probability.shape[:-1]).unsqueeze(1)

                function = torch.cat([function, func_pred], dim=1)
                if function.shape[1] == self.func_res*num_2bar:
                    break
            

        
        function = function.reshape(batch, num_2bar, self.func_res, max_track).permute(0, 1, 3, 2)
        z_func = self.function_encoder.infer_by_codes(function)
        return self.QA_model.infer_with_function_codes(mix[0], prog[0].repeat(num_2bar, 1), z_func[0])
    
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
        vqQaA = Query_and_reArrange(name='pretrain', trf_layers=2, device=DEVICE)
        if pretrain_model_path is not None:
            vqQaA.load_state_dict(torch.load(pretrain_model_path, map_location=torch.device('cpu')))
        vqQaA.eval()
        model = cls(vqQaA.mixture_enc, vqQaA.function_enc, DEVICE=DEVICE).to(DEVICE)
        return model
    
    @classmethod
    def init_inference_model(cls, prior_model_path, QA_model_path, DEVICE='cuda:0'):
        """Fast model initialization."""
        vqQaA = Query_and_reArrange(name='pretrain', trf_layers=2, device=DEVICE)
        vqQaA.load_state_dict(torch.load(QA_model_path, map_location=torch.device('cpu')))
        vqQaA.eval()
        model = cls(inference=True, QA_model=vqQaA, DEVICE=DEVICE).to(DEVICE)
        model.load_state_dict(torch.load(prior_model_path), strict=False)
        return model
    

if __name__ == '__main__':
    from prior_dataset import VQ_LMD_Dataset, collate_fn
    import numpy as np
    import os
    os.environ['CUDA_VISIBLE_DEVICES']= '1'
    from torch.utils.data import DataLoader
    DEVICE = 'cuda:0'
    
    """dataset = VQ_LMD_Dataset(lmd_dir="/data1/zhaojw/LMD/VQ-Q&A-T-009/", debug_mode=True, split='Train', mode='train')
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda b: collate_fn(b, DEVICE))

    model = Prior.init_model(pretrain_model_path="data_file_dir/params_qa.pt", DEVICE=DEVICE)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    for mixture, programs, func_time, track_mask, time_mask, total_length, abs_pos, rel_pos in loader:
        loss = model('loss', mixture, programs, func_time, track_mask, time_mask, total_length, abs_pos, rel_pos)
        print(loss)

        break"""


    prior_model_path = './new_prior.pt'
    QA_model_path = 'data_file_dir/params_qa.pt'
    orchestrator = Prior.init_inference_model(prior_model_path, QA_model_path, DEVICE=DEVICE)
    orchestrator.to(DEVICE)
    orchestrator.eval()
