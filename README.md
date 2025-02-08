# Overview
This repository contains code implementation with **my modifications** for the book [Building LLM from scratch](https://github.com/rasbt/LLMs-from-scratch).

# Notes:
### Chapter 3: `Coding Attention Mechanism`
![alt text](assets/image.png)

> We can improve the SelfAttention_v1 implementation further by utilizing PyTorch’s nn.Linear layers, which effectively perform matrix multiplication when the bias units are disabled. Additionally, a significant advantage of using nn.Linear instead of manually implementing nn.Parameter(torch.rand(...)) is that nn.Linear has an optimized weight initialization scheme, contributing to more stable and effective model training.

> `Causal attention`, also known as `masked attention`, is a specialized form of self-attention. It restricts a model **to only consider previous and current inputs** in a sequence when processing any given token when computing attention scores. **This is in contrast to the standard self-attention mechanism, which allows access to the entire input sequence at once.**

> While all added code lines should be familiar at this point, we now added a `self.register_buffer()` call in the `__init__` method. The use of register_buffer in PyTorch is not strictly necessary for all use cases but offers several advantages here. For instance, when we use the CausalAttention class in our LLM, **buffers are automatically moved to the appropriate device (CPU or GPU) along with our model**, which will be relevant when training our LLM. This means we don’t need to manually ensure these tensors are on the same device as your model parameters, avoiding device mismatch errors.<br><br>
If you have parameters in your model, which should be saved and restored in the `state_dict`, but not trained by the optimizer, you should register them as buffers.
Buffers won’t be returned in `model.parameters()`, so that the optimizer won’t have a change to update them.

> We mask out the attention weights above the diagonal, and we **normalize the nonmasked** attention weights such that the attention weights sum to 1 in each row.

![alt text](assets/image2.png)

> **Information leakage** <br><br>
When we apply a mask and then renormalize the attention weights, **it might initially appear that information from future tokens (which we intend to mask)** could still influence the current token because their values are part of the softmax calculation. However, the key insight is that when we renormalize the attention weights after masking, what we’re essentially doing is recalculating the softmax over a smaller subset (since masked positions don’t contribute to the softmax value). <br><br>
The mathematical elegance of softmax is that despite initially including all positions in the denominator, **after masking and renormalizing, the effect of the masked positions is nullified** they don’t contribute to the softmax score in any meaningful way. <br> <br>
In simpler terms, after masking and renormalization, the distribution of attention weights is as if it was calculated only among the unmasked positions to begin with. **This ensures there’s no information leakage from future (or otherwise masked) tokens as we intended.**

> we **mask with -inf instead of zeros** to properly calculate softmax.

> **Masking additional attention weights with dropout**<br><br>
 It’s important to emphasize that `dropout is only used during training` and is disabled afterward.<br><br>  
Dropout in the attention mechanism is typically applied at two specific times: 
>1. after calculating the attention weights 
>2. or after applying the attention weights to the value vectors.

![alt text](assets/image3.png)

![alt text](assets/image4.png)

> Implementing MultiHead Attention in one matrix multiplication:
![alt text](assets/image5.png)

>Additionally, we added an **output projection layer** (self.out_proj) to `MultiHeadAttention` after combining the heads, which is not present in the CausalAttention class. This output projection layer is not strictly necessary (see appendix B for more details), but it is commonly used in many LLM architectures, which is why I added it here for completeness.

> Logic of making MultiHead Attention in one matrix multiplication independently is that:<br><br>
> * Input_vector shape is `(batch_size, context_length, out_dim)` whereis **out_dim = num_heads * head_dim**
> * We `.view()` this Input_vector of shape `(batch_size, context_length, out_dim)` --> `(batch_size, context_length, num_heads, head_dim)`
> * Then we `.transpose(1,2)` to make the shpae `(batch_size, num_heads, context_length, head_dim)`
> * Then we make the dot product and mask along the last two dimensions ***context_length & num_head***
> * The output **context_vector** now is of shape `(batch_size, num_heads,context_length, head_dim)`
> * Redo the `.transpose(1,2)` we made.
> * Then `.view(batch_size, context_length, out_dim)` to redo the original `.view()`.
>> To determine whether the two operations result in the same vector or just the same shape, let's analyze each operation step by step.
>>  * **First Operation:**<br>
        Start with a vector of shape (2, 64, 10).
        Apply `.view(2, 64, 5, 2)`: This reshapes the vector to (2, 64, 5, 2). The total number of elements remains the same (2 * 64 * 10 = 1280).<br>
        Then apply `.transpose(1, 2)`: This swaps the second and third dimensions, resulting in a shape of (2, 5, 64, 2).
>>  * **Second Operation:**<br>
        Start with the same vector of shape (2, 64, 10).
        Apply `.view(2, 5, 64, 2)`: This also reshapes the vector to (2, 5, 64, 2).
>>  * **Comparison:**<br>
            Both operations result in a tensor of shape (2, 5, 64, 2).
            **However, the contents of the tensors may differ.** The first operation involves a transpose, which changes the order of the elements in the tensor, while the second operation simply reshapes the tensor without changing the order of the elements.

-->  Note:
> logical Order above of .transpose() then .view() is important. 
______________________________________________________

### Chapter 4: Reproducing GPT-2

> torch.nn.Embedding is just a look up table weights.
```python
import torch
vocab_size = 1024
embedding_dim = 768 
emb_layer = torch.nn.Embedding(vocab_size, embedding_dim)
# If I run my input on the randomly generated look up embedding
inp1 = torch.tensor([1,2,3])
emb1 = emb_layer(inp1) # shape: (3, 768) 

# if my input containes number >= 1024 (vocab_size), I will get an error IndexOutOfRange as the following.
inp2 = torch.tensor([1024,2,3])
emb2 = emb_layer(inp2)
```
> The above indicates that, we tokenize our input **(giviing it  number between 0 and vocab_size)** before passing it to embedding layer thus ensuring that our input won't contain out of index numbers.

> Training deep neural networks with many layers can sometimes prove challenging due to problems like **vanishing or exploding gradients**.

> The main idea behind layer normalization is to adjust the activations (outputs) of a neural network layer to have a mean of 0 and a variance of 1. It is normally applied before and after multi-head-attention.

> ![alt text](assets/image6.png)

> This value is very close to 0, but it is not exactly 0 due to small numerical errors that can accumulate because of the finite precision with which computers represent numbers.

> The variable eps is a small constant (epsilon) added to the variance to prevent division by zero during normalization

>  For LLMs, where the embedding dimension n is significantly large, the difference between using n and n – 1 is practically negligible

>  Unlike batch normalization, which normalizes across the batch dimension, layer normalization normalizes across the feature dimension

> Let’s discuss the concept behind shortcut connections, also known as skip or residual connections. Originally, shortcut connections were proposed for deep networks in computer vision (specifically, in residual networks) to mitigate the challenge of vanishing gradients.

> he idea is that the self-attention mechanism in the multi-head attention block identifies and analyzes relationships between elements in the input sequence. In contrast, the feed forward network modifies the data individually at each position

> ![alt text](assets/image7.png)

```python
def generate_text_simple(model, idx,                 #1
                         max_new_tokens, context_size): 
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]    #2
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]                    #3
        probas = torch.softmax(logits, dim=-1)           #4
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)    #5
        idx = torch.cat((idx, idx_next), dim=1)     #6
    return idx
```
> #### #2 Crops current context if it exceeds the supported context size, e.g., if LLM supports only 5 tokens, and the context size is 10, then only the last 5 tokens are used as context 
> **It iterates for a specified number of new tokens to be generated, crops the current context to fit the model’s maximum context size**, computes predictions, and then selects the next token based on the highest probability prediction. <br><br>
> To code the `generate_text_simple function`, we use **a softmax function** to convert the logits into a probability distribution from which we identify the position with the highest value via torch.argmax. The `softmax` function is **monotonic**, meaning it preserves the order of its inputs when transformed into outputs. So, in practice, the softmax step is redundant since the position with the highest score in the softmax output tensor is the same position in the logit tensor. In other words, we could apply the `torch.argmax` function to the logits tensor directly and get identical results. However, I provide the code for the conversion to illustrate the full process of transforming logits to probabilities, which can add additional intuition so that the model generates the most likely next token, which is known as greedy decoding.