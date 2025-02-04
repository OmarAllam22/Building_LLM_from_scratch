# Overview
This repository contains code implementation with **my modifications** for the book [Building LLM from scratch](https://github.com/rasbt/LLMs-from-scratch).

# Notes:

![alt text](assets/image.png)

> We can improve the SelfAttention_v1 implementation further by utilizing PyTorch’s nn.Linear layers, which effectively perform matrix multiplication when the bias units are disabled. Additionally, a significant advantage of using nn.Linear instead of manually implementing nn.Parameter(torch.rand(...)) is that nn.Linear has an optimized weight initialization scheme, contributing to more stable and effective model training.

> `Causal attention`, also known as `masked attention`, is a specialized form of self-attention. It restricts a model **to only consider previous and current inputs** in a sequence when processing any given token when computing attention scores. **This is in contrast to the standard self-attention mechanism, which allows access to the entire input sequence at once.**

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