#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model.py

This file contains the definition of a LLaMA transformer model.
Mostly ported from [karpathy/llama2.c](https://github.com/karpathy/llama2.c)
and [facebookresearch/llama](https://github.com/facebookresearch/llama)

Example usage:
```python
    ckpt = '../_models/stories15M.pt'
    model = get_model_from_ckpt(ckpt)
    model.summary()
    res = model.generate("Once upon a time ", max_new_tokens=50, top_k=40, seed=1234)
    print(res)
```
"""

__author__ = "Abdeladim S."
__copyright__ = "Copyright 2023, "

import math
import os
from pathlib import Path

# os.environ["KERAS_BACKEND"] = "torch"
# os.environ["KERAS_BACKEND"] = "jax"

# Force cpu
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras
from keras import ops
from dataclasses import dataclass
from typing import Optional, Tuple

from llama_lite.tokenizer import Tokenizer
from llama_lite.model import precompute_freqs_cis, reshape_for_broadcast, repeat_kv, FeedForward, RMSNorm, get_model_from_ckpt

@dataclass
class ModelArgs:
    # default hyperparameters for the Llama 7B model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    max_batch_size: int = 32
    dropout: float = 0.0
    use_kv_cacke: bool = True



def apply_rotary_emb(xq: keras.KerasTensor, xk: keras.KerasTensor, freqs_cos: keras.KerasTensor,
                     freqs_sin: keras.KerasTensor, use_kv_cacke: bool = False) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """
    Apply rotary embeddings to q,k tensors
    :param xq: the query tensor
    :param xk: the key tensor
    :param freqs_cos: cos freq
    :param freqs_sin: sin freq
    :return: Tuple of the new (xq,xk) tensors with freqs applied
    """
    # xq_shape = xq.shape[:-1] + (xq.shape[-1] // 2, 2)
    xq_shape = (-1,) + tuple(xq.shape[1:-1]) + (xq.shape[-1] // 2, 2)
    xq = ops.reshape(xq, xq_shape)

    # xk_shape = xk.shape[:-1] + (xk.shape[-1] // 2, 2)
    xk_shape = (-1,) + tuple(xk.shape[1:-1]) + (xk.shape[-1] // 2, 2)
    xk = ops.reshape(xk, xk_shape)

    xq_r, xq_i = ops.unstack(xq, 2, axis=-1)
    xk_r, xk_i = ops.unstack(xk, 2, axis=-1)
    if not use_kv_cacke:
        freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
        freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    xq_out = ops.stack([xq_out_r, xq_out_i], axis=-1)

    # new_shape = tuple(xq_out.shape[:-2]) + (-1,)
    new_shape = (-1,) + tuple(xq_out.shape[1:-2]) + (xq_out.shape[-2] * xq_out.shape[-1],)
    xq_out = ops.reshape(xq_out, new_shape)
    xk_out = ops.stack([xk_out_r, xk_out_i], axis=-1)
    xk_out = ops.reshape(xk_out, new_shape)
    return ops.cast(xq_out, xq.dtype), ops.cast(xk_out, xk.dtype)


class Attention(keras.Model):
    """
    Multi-head attention
    """

    def __init__(self, args: ModelArgs, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = keras.layers.Dense(args.n_heads * self.head_dim, use_bias=False, name='wq')
        self.wk = keras.layers.Dense(self.n_kv_heads * self.head_dim, use_bias=False, name='wk')
        self.wv = keras.layers.Dense(self.n_kv_heads * self.head_dim, use_bias=False, name='wv')
        self.wo = keras.layers.Dense(args.dim, use_bias=False, name='wo')
        self.attn_dropout = keras.layers.Dropout(args.dropout)
        self.resid_dropout = keras.layers.Dropout(args.dropout)
        self.dropout = args.dropout

        # flash attn not supported on all backends
        mask = ops.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"), dtype=self.dtype)
        self.mask = ops.triu(mask, k=1)

    def build(self, input_shape):
        self.wq.build(input_shape)
        self.wk.build(input_shape)
        self.wv.build(input_shape)
        self.wo.build(input_shape)
        self.built = True

    def call(self, x, freqs_cos, freqs_sin, batch_size=1, training=False, kvcache=None, mask_for_kvcache=None, **kwargs):
        bsz, seqlen, _ = x.shape
        # if bsz is None:
        bsz = -1
        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = ops.reshape(xq, (bsz, seqlen, self.n_local_heads, self.head_dim))
        xk = ops.reshape(xk, (bsz, seqlen, self.n_local_kv_heads, self.head_dim))
        xv = ops.reshape(xv, (bsz, seqlen, self.n_local_kv_heads, self.head_dim))
        # RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin, self.args.use_kv_cacke)

        if kvcache:
            old_k, old_v = kvcache
            xk = ops.concatenate([old_k, xk], axis=1)
            xv = ops.concatenate([old_v, xv], axis=1)
        current_cache = [xk[:, 1:, ...], xv[:, 1:, ...]]

        # grouped multiquery attention
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # heads into batch dim
        xq = ops.transpose(xq, (0, 2, 1, 3))
        xk = ops.transpose(xk, (0, 2, 1, 3))
        xv = ops.transpose(xv, (0, 2, 1, 3))

        # attention
        scores = ops.matmul(xq, ops.transpose(xk, (0, 1, 3, 2))) / math.sqrt(self.head_dim)
        if self.args.use_kv_cacke:
            scores += mask_for_kvcache
        else:
            scores += self.mask[:, :, :seqlen, :seqlen]  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = ops.cast(scores, 'float32')
        scores = ops.softmax(scores, axis=-1)
        scores = ops.cast(scores, xq.dtype)
        scores = self.attn_dropout(scores)
        output = ops.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dim and concat heads
        output = ops.transpose(output, (0, 2, 1, 3))
        output = ops.reshape(output, (bsz, seqlen, output.shape[-2] * output.shape[-1]))

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output, current_cache


class TransformerBlock(keras.Model):
    """
    Transformer block
    """

    def __init__(self, layer_id: int, args: ModelArgs, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def build(self, input_shape):
        self.attention.build(input_shape)
        self.feed_forward.build(input_shape)
        self.attention_norm.build(input_shape)
        self.ffn_norm.build(input_shape)
        self.built = True

    def call(self, x, freqs_cos, freqs_sin, batch_size=1, training=False, kvcache=None, mask_for_kvcache=None, **kwargs):
        h, updated_cache = self.attention(self.attention_norm(x), 
                                        freqs_cos, 
                                        freqs_sin, 
                                        batch_size=batch_size,
                                        kvcache=kvcache,
                                        mask_for_kvcache=mask_for_kvcache)
        h = x + h
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, updated_cache


class LLaMATransformer(keras.Model):
    """LLaMA transformer Model """

    def __init__(self, params: ModelArgs, tokenizer_model: str = None, jit_generate=False, **kwargs):
        super().__init__(**kwargs)
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = keras.layers.Embedding(params.vocab_size, params.dim)
        self.dropout = keras.layers.Dropout(params.dropout)
        self.t_layers = [TransformerBlock(layer_id, params, name=f'layer.{layer_id}') for layer_id in
                         range(params.n_layers)]
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.embed_out = keras.layers.Dense(params.vocab_size, use_bias=False, name='output')

        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len)

        seqlen = params.max_seq_len
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        if params.use_kv_cacke:
            freqs_cos = freqs_cos[-1:]
            freqs_sin = freqs_sin[-1:]

            self.freqs_cos_inp = keras.layers.Input(shape=(freqs_cos.shape[-1],), dtype="float32", name="freqs_cos_inp")
            self.freqs_sin_inp = keras.layers.Input(shape=(freqs_sin.shape[-1],), dtype="float32", name="freqs_sin_inp")
            self.mask_inp = keras.layers.Input(shape=(1, 1, seqlen), dtype="float32", name="mask")
            self.tokens_inp = keras.layers.Input(shape=(1,), dtype="int32", name="tokens")

            self.kvcache = []
            n_kv_heads = params.n_heads if params.n_kv_heads is None else params.n_kv_heads
            for i in range(params.n_layers):
                self.kvcache.append(keras.layers.Input(shape=(seqlen - 1, n_kv_heads, params.dim // params.n_heads), dtype="float32", name=f"kcache_{i}"))
                self.kvcache.append(keras.layers.Input(shape=(seqlen - 1, n_kv_heads, params.dim // params.n_heads), dtype="float32", name=f"vcache_{i}"))

            h = self.tok_embeddings(self.tokens_inp) 
            
            self.new_kvcache = []
            self.hidden_inp = []
            for idx, layer in enumerate(self.t_layers):
                kvcache_block = None
                mask = None
                if params.use_kv_cacke:
                    kvcache_block = [self.kvcache[2 * idx], self.kvcache[2 * idx + 1]]
                    mask = self.mask_inp
                
                h, updated_cache = layer(h, self.freqs_cos_inp, self.freqs_sin_inp, kvcache=kvcache_block, mask_for_kvcache=mask)
                self.new_kvcache += updated_cache
                self.hidden_inp.append(h)
            h = self.norm(h)
            self.logits = self.embed_out(h)

            all_output = {"logits": self.logits, "kvcache":ops.concatenate(self.new_kvcache, 1)}

            all_input = {"tokens": self.tokens_inp,
                         "cos": self.freqs_cos_inp,
                         "sin": self.freqs_sin_inp,
                         "mask": self.mask_inp}
            for idx in range(params.n_layers):
                all_input[f"kcache_{idx}"] = self.kvcache[idx * 2]
                all_input[f"vcache_{idx}"] = self.kvcache[idx * 2 + 1]
        
            super().__init__(all_input, all_output)

        else:
            self.tokens_inp = keras.layers.Input(shape=(params.max_seq_len,), dtype="int32", name="tokens")
            h = self.tok_embeddings(self.tokens_inp) 
            h = self.dropout(h)
            for idx, layer in enumerate(self.t_layers):
                h, updated_cache = layer(h, freqs_cos, freqs_sin)
            h = self.norm(h)
            self.logits = self.embed_out(h)
            super().__init__(self.tokens_inp, self.logits)

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None

        if tokenizer_model is not None:
            self.tokenizer = Tokenizer(tokenizer_model)
        else:
            self.tokenizer = Tokenizer(str(Path(__file__).parent.resolve() / "tokenizer.model"))

        self.jit_generate = jit_generate
        # workaround
        self.batch_size = 1

    # def build(self, input_shape):
    #     self.tok_embeddings.build(input_shape)
    #     embeddings_output_shape = self.tok_embeddings.compute_output_shape(input_shape)
    #     for layer in self.t_layers:
    #         layer.build(embeddings_output_shape)
    #     self.norm.build(embeddings_output_shape)
    #     self.embed_out.build(embeddings_output_shape)
    #     self.built = True

    # def call(self, tokens: keras.KerasTensor, targets: keras.KerasTensor = None, batch_size=1, training=False):
    #     if not self.jit_generate:
    #         _bsz, seqlen = tokens.shape
    #     else:
    #         _bsz, seqlen = tokens.shape
    #         if _bsz is None:
    #             _bsz = self.batch_size
    #         # I found that, for the generate function to be tensorflow/jax jit compatible, the seqlen should always be fixed!
    #         # set it to max_seq_len
    #         assert seqlen == self.params.max_seq_len

    #     h = self.tok_embeddings(tokens)
    #     h = self.dropout(h)
    #     freqs_cos = self.freqs_cos[:seqlen]
    #     freqs_sin = self.freqs_sin[:seqlen]

    #     for layer in self.t_layers:
    #         h = layer(h, freqs_cos, freqs_sin, batch_size=_bsz)

    #     h = self.norm(h)

    #     if targets is not None:
    #         # if we are given some desired targets also calculate the loss
    #         logits = self.embed_out(h)
    #         self.last_loss = ops.categorical_crossentropy(ops.reshape(logits, (-1, logits.shape[-1])),
    #                                                       ops.reshape(targets, (-1)), from_logits=True)
    #     else:
    #         # inference-time mini-optimization: only forward the output on the very last position
    #         if not self.jit_generate:
    #             logits = self.embed_out(h[None, :, -1, :])
    #         else:
    #             logits = self.embed_out(h)
    #         self.last_loss = None

    #     return logits

    def load_weights(self, filepath: str, skip_mismatch=False, **kwargs):
        if str(filepath).endswith('pt') or str(filepath).endswith('bin'):
            # load from torch llama2.c models
            import torch
            ckpt = torch.load(filepath)

            self._load(ckpt['model'])
        elif str(filepath).endswith('pth'):
            import torch
            torch.set_default_tensor_type(torch.FloatTensor)
            ckpt = torch.load(filepath)

            self._load(ckpt)
        else:
            # keras/tensorflow weights
            super().load_weights(filepath, skip_mismatch, **kwargs)

    def _load(self, ckpt):
        self.tok_embeddings.set_weights([ckpt['tok_embeddings.weight']])

        def set_transformer_layer(tblock, i):
            tblock.attention.wq.set_weights([ckpt[f'layers.{i}.attention.wq.weight'].T])
            tblock.attention.wk.set_weights([ckpt[f'layers.{i}.attention.wk.weight'].T])
            tblock.attention.wv.set_weights([ckpt[f'layers.{i}.attention.wv.weight'].T])
            tblock.attention.wo.set_weights([ckpt[f'layers.{i}.attention.wo.weight'].T])
            # load ffn
            tblock.feed_forward.w1.set_weights([ckpt[f'layers.{i}.feed_forward.w1.weight'].T])
            tblock.feed_forward.w2.set_weights([ckpt[f'layers.{i}.feed_forward.w2.weight'].T])
            tblock.feed_forward.w3.set_weights([ckpt[f'layers.{i}.feed_forward.w3.weight'].T])
            # load norms
            tblock.attention_norm.set_weights([ckpt[f'layers.{i}.attention_norm.weight']])
            tblock.ffn_norm.set_weights([ckpt[f'layers.{i}.ffn_norm.weight']])

        for i, layer in enumerate(self.t_layers):
            set_transformer_layer(layer, i)

        self.norm.set_weights([ckpt['norm.weight']])
        self.embed_out.set_weights([ckpt['output.weight'].T])


    def sample_next_token(self, current_logits, temp, top_k, seed):
        if keras.backend.backend() != 'jax':
            if temp == 0.:
                _, next_token = ops.top_k(current_logits, k=1)
            else:
                # scale by temp
                current_logits = current_logits / temp
                if top_k is not None:
                    v, _ = ops.top_k(current_logits, k=top_k)
                    current_logits = ops.where(current_logits < v[:, -1], -float("Inf"), current_logits)

                next_token = keras.random.categorical(logits=current_logits, num_samples=1, dtype='int32')

            return ops.cast(next_token, dtype='int32')
        else:
            # jax again!!
            def if_true():
                return ops.top_k(current_logits, k=1)[1]

            def if_false():
                nlogits = current_logits / temp
                if top_k is not None:
                    v, _ = ops.top_k(nlogits, min(top_k, nlogits.shape[-1]))
                    nlogits = ops.where(nlogits < v[:, -1], -float("Inf"), nlogits)
                    # pass

                # no random keys in this world again!!
                # keras_core api throw error with jax/tracer seed generator
                # Couldn't find a way to make it random without breaking the XLA
                import jax
                key = jax.random.PRNGKey(0)
                output_shape = list(nlogits.shape)
                output_shape[1] = 1
                output_shape = tuple(output_shape)
                output = jax.random.categorical(
                    key=key, logits=nlogits[..., None], shape=output_shape, axis=1
                )
                return output.astype('int32')

            return ops.cond(temp == 0, if_true, if_false)
    
    def _get_generate(self):
        def _eager_generate(tokens, max_new_tokens, temp=0.0, top_k=None, seed=None):
            for _ in range(max_new_tokens):
                # crop if too long
                idx_cond = tokens if tokens.shape[1] <= self.params.max_seq_len else tokens[:, -self.params.max_seq_len:]
                # forward
                logits = self.call(idx_cond, training=False)
                logits = logits[:, -1, :]
                if temp == 0.:
                    _, next_token = ops.top_k(logits, k=1)
                else:
                    # scale by temp
                    logits = logits / temp
                    if top_k is not None:
                        v, _ = ops.top_k(logits, min(top_k, logits.shape[-1]))
                        logits = ops.where(logits < v[:, -1], -float("Inf"), logits)
                    next_token = keras.random.categorical(logits=logits, num_samples=1, seed=seed)
                # append sampled idx to the running seq and continue
                tokens = ops.concatenate([tokens, next_token], axis=1)

            return tokens

        def _jit_able_generate(tokens, max_new_tokens, temp=0.0, top_k=None, seed=None, start_at_token=None):
            """
            A helper function to be able to jit the generate function, in the hope of gaining a performance boost

            :param tokens:
            :param max_new_tokens:
            :param temp:
            :param top_k:
            :param seed:
            :return:
            """
            bsz, seq_len = tokens.shape
            if start_at_token is not None:
                seq_len = start_at_token
                output = tokens
            else:
                # pad tokens to max_seq_len
                output = ops.zeros((1, self.params.max_seq_len - seq_len), dtype='int32')
                output = ops.concatenate([tokens, output], axis=-1)

            def loop(i, output):
                logits = self.call(output, training=False)
                # get current token logits
                current = seq_len + i - 1
                logits = logits[:, current, :]
                next_token = self.sample_next_token(logits, temp, top_k, seed)
                # update output, here as well!!!
                if keras.backend.backend() != 'jax':
                    output = ops.scatter_update(output, indices=[[0, seq_len + i]], updates=next_token[0])
                else:
                    output = output.at[0, seq_len + i].set(next_token[0][0])
                return output

            # run fori
            output = ops.fori_loop(0, max_new_tokens, loop, output)
            return output[:, :max_new_tokens]

        if self.jit_generate:
            return _jit_able_generate
        else:
            return _eager_generate
    
    def generate(self, prompt: str, max_new_tokens, temp=0.0, top_k=None, seed=None):

        tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        tokens = ops.convert_to_tensor(tokens, dtype='int32')
        tokens = ops.expand_dims(tokens, axis=0)  # add batch dim

        _fnc = self._get_generate()
        if self.jit_generate:
            # print(f"JIT generate: {self.jit_generate}")
            # torch is very fast even without jit

            if keras.backend.backend() == 'tensorflow':
                import tensorflow as tf
                _fnc = tf.function(_fnc)

            elif keras.backend.backend() == 'jax':
                import jax
                _fnc = jax.jit(_fnc, static_argnames=['max_new_tokens', 'top_k'])

        res = _fnc(tokens, max_new_tokens, temp, top_k, seed)
        res = self.tokenizer.decode(ops.convert_to_numpy(res).tolist())
        return res[0]



if __name__ == '__main__':
    ckpt = '../_models/stories15M.pt'
    model = get_model_from_ckpt(ckpt, jit_generate=True)
    model.summary()
    res = model.generate("Once upon a time ", max_new_tokens=10, temp=0.8, top_k=15, seed=1234)
    print(res)
