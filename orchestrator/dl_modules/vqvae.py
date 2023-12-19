import torch
from torch import nn
import torch.nn.functional as F

class VectorQuantizerEMA(nn.Module):
    """
    Discretization bottleneck of VQ-VAE using EMA with random restart.
    After certain iterations, run:
    random_restart()
    reset_usage()
    """
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, decay, usage_threshold, epsilon=1e-5, random_start=False):
        super(VectorQuantizerEMA, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.decay = decay
        self.commitment_cost = commitment_cost
        self.usage_threshold = usage_threshold
        self.epsilon = epsilon
        self.random_start = random_start

        with torch.no_grad():
            self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
            #self.embedding.weight.data.normal_()
            self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
            self.register_buffer('usage', torch.ones(self.num_embeddings), persistent=False)
            self.register_buffer('ema_cluster_size', torch.zeros(self.num_embeddings), persistent=False)
            self.register_buffer('ema_w', self.embedding.weight.data.clone(), persistent=False)

        self.perplexity = None
        self.loss = None

    def update_usage(self, min_enc):
        with torch.no_grad():
            self.usage[min_enc] = self.usage[min_enc] + 1  # if code is used add 1 to usage
            self.usage /= 2 # decay all codes usage

    def reset_usage(self):
        with torch.no_grad():
            self.usage.zero_() #  reset usage between certain numbers of iterations

    def random_restart(self, batch_z=None):
        #  randomly restart all dead codes below threshold with random code from the batch
        with torch.no_grad():
            mean_usage = torch.mean(self.usage[self.usage >= self.usage_threshold])
            dead_codes = torch.nonzero(self.usage < self.usage_threshold).squeeze(1)
            if self.random_start:
                if batch_z is None:
                    rand_codes = torch.randperm(self.num_embeddings)[0:len(dead_codes)]
                    self.embedding.weight[dead_codes] = self.embedding.weight[rand_codes]
                    self.ema_w[dead_codes] = self.embedding.weight[rand_codes]
                else: 
                    LEN = min(len(dead_codes), len(batch_z))
                    rand_codes = torch.randperm(len(batch_z))[0:LEN]
                    self.embedding.weight[dead_codes[0:LEN]] = batch_z[rand_codes]
                    self.ema_w[dead_codes[0:LEN]] = batch_z[rand_codes]
            return mean_usage, len(dead_codes)

    def forward(self, z, track_pad_mask=None):
        #z shape: (batch*max_track, embedding_dim)
        #track_pad_mask: (batch, max_track)
        assert(z.shape[-1] == self.embedding_dim)
        input_shape = z.shape
        z = z.reshape(-1, z.shape[-1])
        track_pad_mask = track_pad_mask.reshape(-1)

        distance = torch.sum(z ** 2, dim=1, keepdim=True) \
           + torch.sum(self.embedding.weight ** 2, dim=1) \
           - 2 * torch.matmul(z, self.embedding.weight.t()) #(batch*max_track, num_embeddings)
        
        min_encoding_indices = torch.argmin(distance, dim=1)   #(batch*max_track,)
        #print(min_encoding_indices)
        min_encodings = torch.zeros(len(min_encoding_indices), self.num_embeddings, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)  #(batch*max_track, num_embeddings)

        z_q = torch.matmul(min_encodings, self.embedding.weight)    #(batch*max_track, embedding_dim)
        
        self.update_usage(min_encoding_indices[torch.logical_not(track_pad_mask)])

        if self.training:
            with torch.no_grad():
                self.ema_cluster_size -= (1 - self.decay) * (self.ema_cluster_size - torch.sum(min_encodings[torch.logical_not(track_pad_mask)], dim=0))
                #laplacian smoothing
                n = torch.sum(self.ema_cluster_size.data)
                self.ema_cluster_size = (self.ema_cluster_size + self.epsilon) * n \
                                        / (n + self.num_embeddings * self.epsilon)

                dw = torch.matmul(min_encodings[torch.logical_not(track_pad_mask)].t(), z[torch.logical_not(track_pad_mask)])  #(num_embeddings, embed_dim)
                self.ema_w -= (1-self.decay) * (self.ema_w - dw)
                self.embedding.weight.data = self.ema_w / self.ema_cluster_size.unsqueeze(-1)

        e_latent_loss = F.mse_loss(z_q.detach(), z)
        loss = self.commitment_cost * e_latent_loss

        quantized = (z + (z_q - z).detach()).reshape(input_shape)
        avg_probs = torch.mean(min_encodings[torch.logical_not(track_pad_mask)], dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity
    
    def get_code_indices(self, z):
        assert(z.shape[-1] == self.embedding_dim)
        input_shape = z.shape
        z = z.reshape(-1, z.shape[-1])

        distance = torch.sum(z ** 2, dim=1, keepdim=True) \
           + torch.sum(self.embedding.weight ** 2, dim=1) \
           - 2 * torch.matmul(z, self.embedding.weight.t()) #(batch*max_track, num_embeddings)
        
        min_encoding_indices = torch.argmin(distance, dim=1)   #(batch*max_track,)
        return min_encoding_indices.reshape(input_shape[:-1])
    
    def infer_code(self, encoding_indices):
        input_shape = encoding_indices.shape
        encoding_indices = encoding_indices.reshape(-1)
        encodings = torch.zeros(len(encoding_indices), self.num_embeddings, device=encoding_indices.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)  #(batch*max_track, num_embeddings)
        z_q = torch.matmul(encodings, self.embedding.weight)
        return z_q.reshape(*list(input_shape), self.embedding_dim)




class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck of VQ-VAE with random restart.
    After certain iterations, run:
    random_restart()
    reset_usage()
    """
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, usage_threshold, random_start=False):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.usage_threshold = usage_threshold
        self.random_start = random_start

        with torch.no_grad():
            self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
            #self.embedding.weight.data.normal_()
            self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
            self.register_buffer('usage', torch.ones(self.num_embeddings), persistent=False)

        self.perplexity = None
        self.loss = None

    def update_usage(self, min_enc):
        with torch.no_grad():
            self.usage[min_enc] = self.usage[min_enc] + 1  # if code is used add 1 to usage
            self.usage /= 2 # decay all codes usage

    def reset_usage(self):
        with torch.no_grad():
            self.usage.zero_() #  reset usage between certain numbers of iterations

    def random_restart(self):
        #  randomly restart all dead codes below threshold with random code from the codebook
        with torch.no_grad():
            dead_codes = torch.nonzero(self.usage < self.usage_threshold).squeeze(1)
            if self.random_start:
                rand_codes = torch.randperm(self.num_embeddings)[0:len(dead_codes)]
                self.embedding.weight[dead_codes] = self.embedding.weight[rand_codes]
            return len(dead_codes)

    def forward(self, z, track_pad_mask=None):
        #z shape: (batch, max_track, embedding_dim)
        #track_pad_mask: (batch, max_track)
        assert(z.shape[-1] == self.embedding_dim)
        input_shape = z.shape
        z = z.reshape(-1, z.shape[-1])
        track_pad_mask = track_pad_mask.reshape(-1)

        distance = torch.sum(z ** 2, dim=1, keepdim=True) \
           + torch.sum(self.embedding.weight ** 2, dim=1) \
           - 2 * torch.matmul(z, self.embedding.weight.t()) #(batch*max_track, num_embeddings)
        
        min_encoding_indices = torch.argmin(distance, dim=1)   #(batch*max_track,)
        #print(min_encoding_indices)
        min_encodings = torch.zeros(len(min_encoding_indices), self.num_embeddings, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)  #(batch*max_track, num_embeddings)

        z_q = torch.matmul(min_encodings, self.embedding.weight)    #(batch*max_track, embedding_dim)
        
        self.update_usage(min_encoding_indices[torch.logical_not(track_pad_mask)])

        e_latent_loss = F.mse_loss(z_q.detach(), z)
        q_latent_loss = F.mse_loss(z_q, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = (z + (z_q - z).detach()).reshape(input_shape)
        avg_probs = torch.mean(min_encodings[torch.logical_not(track_pad_mask)], dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity

        
        
            
            
        