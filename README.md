# Overview
This repository contains code implementation with **my modifications** for the book [Building LLM from scratch](https://github.com/rasbt/LLMs-from-scratch).

# Notes:

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

### Note:
> logical Order above of .transpose() then .view() is important. 
