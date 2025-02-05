import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout_percentage: float, bias_included: bool = False):
        super().__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.queries = nn.Linear(in_dim, out_dim, bias=bias_included)
        self.keys = nn.Linear(in_dim, out_dim, bias=bias_included)
        self.values = nn.Linear(in_dim, out_dim, bias=bias_included)
        self.dropout = nn.Dropout(p=dropout_percentage)

    def forward(self, x):
        self.q_w = self.queries(x)  # shape (batch_size, context_length, out_dim)
        self.k_w = self.keys(x)
        self.v_w = self.values(x)

        
        self.attn_scores = self.q_w @ self.k_w.transpose(1,2)  # result of shape: (batch_size, context_length, num_tokens)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(x.shape[1], x.shape[1]),diagonal=1).bool()    # x.shape[1] is the context_length or num_tokens
        )
        self.attn_scores.masked_fill_(self.mask, -torch.inf)  # underscore in .masked_fill_() indicates inplace edits.
        
        self.attn_weights = torch.softmax(self.attn_scores / self.q_w.shape[-1]**0.5 , dim = -1)
        self.attn_weights = self.dropout(self.attn_weights)

        self.context_vector = self.attn_weights @ self.v_w 
        return self.context_vector  # result of shape: (batch_size, num_tokens, out_dim) 

# example usage:

c = CausalAttention(3,2,dropout_percentage=0.5)
torch.manual_seed(222)
c(torch.rand(2,3).view(1,2,3)) 
print(c.attn_weights)
