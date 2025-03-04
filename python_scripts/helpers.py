import tiktoken, torch

GPT_CONFIG_124M = GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,    #1
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12, 
    "drop_rate": 0.1,       #2
    "qkv_bias": False
}
tokenizer = tiktoken.get_encoding('gpt2')

def Text2TokenIds(text, tokenizer):
    tensor_ids_unsqueezed = torch.tensor(tokenizer.encode(text, allowed_special={'<|endoftext|>'})).unsqueeze(0)
    return tensor_ids_unsqueezed

def TokenIds2Text(tensor_ids_unsqueezed, tokenizer):
    text = tokenizer.decode(tensor_ids_unsqueezed.squeeze(0).tolist())
    return text

def generate_text_ids_simple(
        model,
        current_ids,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M['context_length']
):
    for i in range(max_new_tokens):
        ids_trimmed_tensor = current_ids[:, -context_size:]  # ids_trimmed_tensor --> (batch, num_tokens)
        with torch.no_grad():                          
            logits = model(ids_trimmed_tensor) 
        last_token_logits = torch.softmax(logits[:,-1,:], axis=-1)   # last_token_logits --> (batch, vocab_size)
        last_token_idx_batched = torch.argmax(last_token_logits, dim=-1, keepdim=True) # last_token_idx_batched --> (batch, 1)
        current_ids = torch.cat((current_ids, last_token_idx_batched), dim=1) #current_ids --> (batch, num_tokens + 1)
    return current_ids

