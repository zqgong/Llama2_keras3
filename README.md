## llama2 keras2

This respository is a Multi-Backend (Pytorch, Tensorflow, Jax) implementation of [LLaMA](https://github.com/facebookresearch/llama) using Keras3. 

Base on [LLaMA-Lite](https://github.com/abdeladim-s/llama-lite).

Implement the KVCache in simple code. [Speed up the GPT](https://www.dipkumar.dev/becoming-the-unbeatable/posts/gpt-kvcache/)


## Inference

* Get the `tinyllama` model weights from [HF](https://huggingface.co/karpathy/tinyllamas/tree/main). 
* You can also try the Llama2 weights from [Meta HF](https://huggingface.co/meta-llama)

```python
python performence.py
```
