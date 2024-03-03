#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple script to compare the performance between the three backends
"""
import subprocess

ckpt = './stories15M.pt'
prompt = "Once upon a time"
max_new_tokens = 50
top_k = 40
seed = 1234
backends = ['torch', 'tensorflow', 'jax']
num_iterations = 1
jit_generate = True
use_cpu = False


def test():
    # A simple program to load the model and run a simple generate of `max_new_tokens` new tokens
    import os
    if use_cpu:
        # Force cpu
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    from llama_lite.model import get_model_from_ckpt
    model = get_model_from_ckpt(ckpt, jit_generate=jit_generate)
    res = model.generate("Once upon a time ", max_new_tokens=max_new_tokens, top_k=top_k, seed=seed)
    print(res)


def run(ckpt=ckpt,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        seed=seed,
        backends=backends,
        num_iterations=num_iterations,
        use_cpu=use_cpu
        ):
    if use_cpu:
        print("Using CPU ...")
    for backend in backends:
        print(f"Running {backend} backend in isolated subprocess ...")
        stm = f"'test()', setup='from performance import test', number={num_iterations}"
        cmd = f'KERAS_BACKEND={backend} python3 -c "import timeit; print(timeit.timeit({stm}))"'
        res = subprocess.check_output([cmd], shell=True)
        time = float(str(res).split('\\n')[-2])
        print(f"Total inference time of generating {max_new_tokens} tokens is {time:.2f} s")


def test_kvcache():
    import os
    if use_cpu:
        # Force cpu
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    from llama_lite.model_kvcache import LLaMATransformer, ModelArgs
    import torch
    import numpy as np

    model_ckpt = torch.load(ckpt)
    model_args = ModelArgs(**model_ckpt['model_args'])
    model_args.use_kv_cacke = True
    model_args.n_kv_heads = model_args.n_kv_heads if model_args.n_kv_heads is not None else model_args.n_heads
    print(model_args)

    model = LLaMATransformer(model_args, tokenizer_model=None, jit_generate=jit_generate)
    model.load_weights(ckpt)

    inp = model.tokenizer.encode("Once upon a time ", bos=True, eos=False)
    mask = np.array([-10000000] * (model_args.max_seq_len - 1) + [0], dtype="float32")

    input_dict = {}
    token = inp[0]
    all_tokens = [1]
    for i in range(max_new_tokens + len(inp)):
        freq_c, freq_s = model.freqs_cos[i: i+1], model.freqs_sin[i: i+1]
        input_dict["tokens"] = np.array([token])
        input_dict["cos"] = np.array(freq_c)
        input_dict["sin"] = np.array(freq_s)
        input_dict["mask"] = np.reshape(mask, (1, 1, 1, -1))

        if i == 0:
            kvcache = [np.zeros((1, model_args.max_seq_len - 1, 
                                    model_args.n_kv_heads, 
                                    model_args.dim // model_args.n_heads))] * model_args.n_layers * 2
            for i in range(model_args.n_layers):
                input_dict[f"kcache_{i}"] = kvcache[i * 2]
                input_dict[f"vcache_{i}"] = kvcache[i * 2 + 1]
        else:
            mask[-i - 1: ] = 0
            input_dict["mask"] = np.reshape(mask, (1, 1, 1, -1))
            for i in range(model_args.n_layers):
                gap = model_args.max_seq_len - 1
                input_dict[f"kcache_{i}"] = np.array(oups["kvcache"][:, gap * (i * 2): gap * (i * 2 + 1)])
                input_dict[f"vcache_{i}"] = np.array(oups["kvcache"][:, gap * (i * 2 + 1): gap * (i * 2 + 2)])
        oups = model(input_dict)

        if i < len(inp) - 1:
            token = inp[i + 1]
        else:
            logits = oups["logits"]
            token = np.array(model.sample_next_token(logits[0], 1, top_k, seed))[0][0]

        all_tokens.append(token)

    res = model.tokenizer.decode(np.array(all_tokens).tolist())
    print(res) 

if __name__ == '__main__':
    test_kvcache()
    # run()
    # test()
