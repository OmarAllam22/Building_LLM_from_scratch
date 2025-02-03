import torch
import torch.nn as nn

class SelfAttentionV1(nn.Module):
    def __init__(self, in_dim, out_dim, bias_include=False):
        super().__init__()
        self.queries_L = nn.Linear(in_dim, out_dim, bias = bias_include)
        self.keys_L = nn.Linear(in_dim, out_dim, bias = bias_include)
        self.values_L = nn.Linear(in_dim, out_dim, bias = bias_include)

        self.queries_P = nn.Parameter(torch.rand(in_dim, out_dim))
        self.keys_P = nn.Parameter(torch.rand(in_dim, out_dim))
        self.values_P = nn.Parameter(torch.rand(in_dim, out_dim))


    def forward(self, x):
        
        query_w = x @ self.keys_P
        key_w = x @ self.queries_P
        value_w = x @ self.values_P

        # or using nn.Linear which implicitly do matrix multiplication
        
        query_w = self.keys_L(x)
        key_w = self.queries_L(x)
        value_w = self.values_L(x)

        # calculating attention scores
        attention_scores = query_w @ key_w.T

        # calculating attention weights
        attention_weights = torch.softmax(
            attention_scores / (key_w.shape[-1])**0.5
            ,dim=-1
        )

        self_attention_context_vector = attention_weights @ value_w
        return self_attention_context_vector

# example usage:
attention_obj = SelfAttentionV1(3,512)
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
context_vector = attention_obj(inputs)
context_vector
