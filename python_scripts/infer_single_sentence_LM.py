import torch

def infer_language_model(model, tokenizer, sentence:str, max_new_tokens:int, context_size:int):
    encoded_sentence = tokenizer.encode(sentence)
    for i in range(max_new_tokens):        
        sentence_idx_tensor = torch.tensor(encoded_sentence[-context_size:]).unsqueeze(0)
        with torch.no_grad():
            logits = model(sentence_idx_tensor)
        last_token_logits = logits[:, -1, :]
        last_token_probas = torch.softmax(last_token_logits, dim=-1)
        last_token_idx = torch.argmax(last_token_probas, dim=-1) # must keepdim=True to stacking the idx to sentence_idx_tensor
        encoded_sentence.append(last_token_idx)
        yield tokenizer.decode(encoded_sentence)
