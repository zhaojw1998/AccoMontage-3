import math
import torch 
import torch.nn.functional as F 
from torch import nn
from torch.nn.modules.normalization import LayerNorm


class MultiheadSelfAttentionwithRelativePositionalEmbedding(nn.Module):
    def __init__(self, dmodel, num_heads, dropout=0, max_len=1024):
        super(MultiheadSelfAttentionwithRelativePositionalEmbedding, self).__init__()
        self.max_len = max_len
        self.num_heads = num_heads
        self.head_dim = dmodel // num_heads
        assert self.head_dim * num_heads == dmodel, "embed_dim must be divisible by num_heads"

        self.key = nn.Linear(dmodel, dmodel)
        self.value = nn.Linear(dmodel, dmodel)
        self.query = nn.Linear(dmodel, dmodel)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(num_heads, (2*max_len-1) + 2, self.head_dim))

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        #x: (batch, len, dmodel)
        #Srel: (num_head, src_len, src_len)
        #key_padding_mask: (batch, src_len), bool tensor
        #attn_mask:  (batch, num_head, src_len, src_len): float tensor
        bs, src_len, d_model = query.shape
        #_, src_len, _ = key.shape

        q = self.query(query).reshape(bs, src_len, self.num_heads, self.head_dim).transpose(1, 2)  #(batch, num_head, src_len, head_dim)
        k = self.key(key).reshape(bs, src_len, self.num_heads, self.head_dim).permute(0, 2, 3, 1)  #(batch, num_head, head_dim, src_len)
        v = self.value(value).reshape(bs, src_len, self.num_heads, self.head_dim).transpose(1, 2)  #(batch, num_head, src_len, head_dim)

        Er_t = self.Er[:, max(0, self.max_len-src_len): min(2*self.max_len-1, self.max_len+src_len-1), :]
        if src_len > self.max_len:
            Er_t = torch.cat([
                self.Er[:, -2, :].unsqueeze(1).repeat(1, src_len-self.max_len, 1),
                Er_t,
                self.Er[:, -1, :].unsqueeze(1).repeat(1, src_len-self.max_len, 1)
            ], dim=1)
        Er_t = Er_t.transpose(-2, -1)   #(num_head, head_dim, 2*src_len-1)

        QEr = torch.matmul(q, Er_t) #(batch, num_head, src_len, 2*src_len-1)
        Srel = self.skew(QEr) #(batch, num_head, src_len, src_len)

        if key_padding_mask is not None:
            if attn_mask is not None:
                attn_mask = attn_mask.masked_fill(key_padding_mask.reshape(bs, 1, 1, src_len), float("-inf"))
            else:
                attn_mask = torch.zeros(bs, 1, 1, src_len, dtype=torch.float).to(key_padding_mask.device)
                attn_mask = attn_mask.masked_fill(key_padding_mask.reshape(bs, 1, 1, src_len), float("-inf"))

        attn = (torch.matmul(q, k) + Srel) / math.sqrt(self.head_dim) #(batch, num_head, tgt_len, src_len)
        
        if attn_mask is not None:
            attn += attn_mask
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v) #(batch, num_head, tgt_len, head_dim)
        out = out.transpose(1, 2).reshape(bs, src_len, d_model) #(batch, tgt_len, d_model)
        return self.dropout(out), attn
    
        
    def skew(self, QEr):
        #QEr: (batch, num_heads, src_len, 2*src_len-1)
        bs, num_heads, src_len, L = QEr.shape
        QEr = F.pad(QEr, (0, 1))    #(batch, num_heads, src_len, L+1)
        QEr = QEr.reshape(bs, num_heads, -1)   #(batch, num_heads, src_len*(L+1))
        QEr = F.pad(QEr, (0, L-src_len))    #(batch, num_heads, (src_len+1)*L)
        QEr = QEr.reshape(bs, num_heads, src_len+1, L)
        QEr = QEr[:, :, :src_len, -src_len:]    #(batch, num_heads, src_len, src_len)
        return QEr



class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5, norm_first=False, max_len=1024):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadSelfAttentionwithRelativePositionalEmbedding(d_model, nhead, dropout, max_len)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        #src: (batch, len, dmodel)
        #key_padding_mask: (batch, src_len), bool tensor
        #attn_mask:  (batch, num_head, src_len, src_len): float tensor
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x, attn_mask=None, key_padding_mask=None):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
