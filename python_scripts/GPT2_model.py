import torch
import torch.nn as nn

from multi_head_causal_scaled_dot_product_self_attention import MultiHeadAttention 
class TransformerBasicBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg['emb_dim']) 
        self.mha = MultiHeadAttention(in_dim = cfg['emb_dim'], out_dim = cfg['emb_dim'],
                                      dropout_percentage = cfg['drop_rate'], bias_included = cfg['qkv_bias'],
                                      num_heads = cfg['n_heads']
        )
        self.dropout = nn.Dropout(cfg['drop_rate'])
        self.ff = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4*cfg['emb_dim']),
            nn.GELU(),
            nn.Linear(4*cfg['emb_dim'], cfg['emb_dim'])
        )
        self.layer_norm2 = nn.LayerNorm(cfg['emb_dim'])            

    def forward(self, x):
        shortcut = x
        x = self.layer_norm1(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = x + shortcut if shortcut.shape == x.shape else x 
        shortcut = x
        x = self.layer_norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut if shortcut.shape == x.shape else x
        return x


class GPT2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(num_embeddings = cfg['vocab_size'], embedding_dim = cfg['emb_dim'])
        self.pos_emb = nn.Embedding(num_embeddings = cfg['context_length'], embedding_dim = cfg['emb_dim'])
        self.dropout = nn.Dropout(cfg['drop_rate'])
        self.repeated_transformer_blocks = nn.Sequential(
            *[TransformerBasicBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        self.final_layer_norm = nn.LayerNorm(cfg['emb_dim'])
        self.out_layer = nn.Linear(cfg['emb_dim'],cfg['vocab_size'], bias=False) # bias equals false, to make weights dimension the same as nn.Embedding to use `weight_tying` if needed

    def forward(self, x):
        x = self.token_emb(x) + self.pos_emb(torch.arange(x.shape[-1]))
        x = self.dropout(x)
        x = self.repeated_transformer_blocks(x)
        x = self.final_layer_norm(x)
        logits = self.out_layer(x)
        return logits
    
    def logits2text(self, logits, tokenizer):
        last_token_logits = logits[ : , -1 ,  : ]
        indices = torch.argmax(torch.softmax(last_token_logits, axis=-1),dim=-1, keepdim=True)

        return indices
    
