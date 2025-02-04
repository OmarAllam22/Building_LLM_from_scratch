import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout_percentage: float, bias_included: bool = False):
        super().__init__()
        self.queries = nn.Linear(in_dim, out_dim, bias=bias_included)
        self.keys = nn.Linear(in_dim, out_dim, bias=bias_included)
        self.values = nn.Linear(in_dim, out_dim, bias=bias_included)
        self.dropout = nn.Dropout(p=dropout_percentage)

    def forward(self, x):
        q_w = self.queries(x)  # shape (batch_size, context_length, out_dim)
        k_w = self.keys(x)
        v_w = self.values(x)

        
        attn_scores = q_w @ k_w.transpose(1,2)  # result of shape: (batch_size, context_length, num_tokens)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(x.shape[1], x.shape[1]),diagonal=1).bool()    # x.shape[1] is the context_length or num_tokens
        )
        attn_scores = attn_scores.masked_fill_(self.mask, -torch.inf)
        
        self.attn_weights = torch.softmax(attn_scores / q_w.shape[-1]**0.5 , dim = -1)
        self.attn_weights = self.dropout(self.attn_weights)

        self.context_vector = self.attn_weights @ v_w 
        return self.context_vector  # result of shape: (batch_size, num_tokens, out_dim) 

# example usage:

c = CausalAttention(3,2,dropout_percentage=0.5)
torch.manual_seed(222)
c(torch.rand(2,3).view(1,2,3)) 
print(c.attn_weights)
