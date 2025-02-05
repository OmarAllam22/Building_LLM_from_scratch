import torch 
import torch.nn as nn
from python_scripts.causal_scaled_dot_product_self_attention import CausalAttention

class MultiHeadAttention(CausalAttention):
    def __init__(self, *args, num_heads=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        assert self.out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.head_dim = self.out_dim // num_heads
        self.out_proj = nn.Linear(self.out_dim, self.out_dim)  # output projection layer
        
    def forward(self, x):
        self.q_w = self.queries(x).view(x.shape[0], self.num_heads ,x.shape[1], self.head_dim) # self.queries(x) result of shape: (batch_size, num_tokens, d_out) ... d_out = head_dim * num_heads ... we just .view() this shape
        self.k_w = self.keys(x).view(x.shape[0], self.num_heads ,x.shape[1], self.head_dim)
        self.v_w = self.values(x).view(x.shape[0], self.num_heads ,x.shape[1], self.head_dim)

        self.attn_scores = self.q_w @ self.k_w.transpose(2,3)   # self.q_w, self.k_w of shape : (batch_size, num_heads , num_tokens, head_dim) ... d_out = num_heads * head_dim
                                                                # self.attn_scores of shape : (batch_size, num_heads , num_tokens, num_tokens)    
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(x.shape[1], x.shape[1]),diagonal=1).bool()   # x.shape[1] is the context_length or num_tokens
        )
        self.attn_scores.masked_fill_(self.mask[:x.shape[1],:x.shape[1]], -torch.inf)  # Note self.mask doesn't change from the definition of parent class CausalAttention.
                                                  # but we didn't run super().forward() in which self.mask is registered ... we ran only super().__init__() 

        self.attn_weights = torch.softmax(self.attn_scores / self.head_dim**0.5 , dim = -1) # self.attn_weights of shape: (batch_size, num_heads, num_tokens, num_tokens)

        self.context_vector = (self.attn_weights @ self.v_w).transpose(1,2)                     # result of self.context_vector BEFORE transpose is of shape: (batch_size, num_heads, num_tokens, head_dim)
                                                                                                # we transpose num_heads with num_tokens ...in order to merge num_heads with head_dim into one dimension (out_dim)

        self.context_vector = self.context_vector.contiguous().view(x.shape[0], x.shape[1], self.out_dim)    # self.context_vector BEFORE .view() was of shape: (batch_size, num_heads, num_tokens, head_dim)
        return self.out_proj(self.context_vector)  # result of shape: (batch_size, num_tokens, out_dim) 

# Example Usage:        
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

batch = torch.stack((inputs, inputs), dim=0)
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2

m = MultiHeadAttention(in_dim=d_in, out_dim=d_out, dropout_percentage=0.0, bias_included=False, num_heads=2)
context_vec = m(batch) 
print(context_vec, context_vec.shape)
