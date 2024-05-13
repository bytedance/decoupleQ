"""
The code in this file is built on thr top of OPTQ, please visit:
https://github.com/IST-DASLab/gptq
for their origin contribution

SPDX-License-Identifier: Apache-2.0

This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (2024) Bytedance Ltd. and/or its affiliates.
"""
import time
import os
from tokenize import group
from xml.sax.handler import feature_external_ges
import torch
import torch.nn as nn
from decoupleQ.quant import decoupleQ, minimize_block
from decoupleQ.moq_quant import Quantizer
from decoupleQ.quant import find_layers, to_device
import shutil
import gc
from decoupleQ.linear_w2a16 import LinearW2A16, LinearA16


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


@torch.no_grad()
def quant_sequential(args, model, layers, dataloader, dev):
    print(args)
    print("start quant====")
    cache = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            inputs = [list(args), kwargs]
            cache.append(to_device(inputs, "cpu"))
            raise ValueError

    layers[0] = Catcher(layers[0])

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)

    # model = model.to(dev)
    model.eval()
    torch.cuda.empty_cache()
    model.requires_grad_(False)
    masks = [None] * len(dataloader)

    for batch in dataloader:
        batch = to_device(batch, dev)
        try:
            model(batch)
        except ValueError:
            pass

    del dataloader, batch
    gc.collect()
    layers[0] = layers[0].module
    model = model.cpu()
    inps = cache
    torch.cuda.empty_cache()

    print('Ready.')
    shift = 0
    quantizers = {}
    outs = []

    for i in range(len(layers)):
        t_layer0 = time.time()
        layer = layers[i]
        full = find_layers(layer)
        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]

        for k, names in enumerate(sequential):
            subset = {n: full[n] for n in names}
            moq = {}
            for name in subset:
                moq[name] = decoupleQ(subset[name], name=f"layer.{i}.{name}")
                moq[name].quantizer = Quantizer()
                moq[name].quantizer.configure(args.qbits, perchannel=True, sym=not args.asym)
                subset[name].mask = [None]

            def add_batch(name):
                def tmp(module, inp, out):
                    moq[name].add_batch(inp[0].data, out.data, module.mask[0])

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            layer = layer.to(dev)
            for idx, b in enumerate(inps):
                b = to_device(b, dev)
                out = layer(*(b[0]), **b[1])
                if k == 0 and args.blockwise_minimize_lr > 0:
                    os.makedirs("./tmp_blockwise", exist_ok=True)
                    out = {"out": to_device(out, "cpu")}
                    torch.save(out, f"./tmp_blockwise/out_{idx}.pth")
                del out
            layer = layer.cpu()

            for h in handles:
                h.remove()

            for name in names:
                del subset[name].mask
                print(i, name)
                print('Quantizing ...')
                t1 = time.time()
                torch.cuda.empty_cache()
                scale_out, zero_out, w_int, loss = moq[name].startquant(
                    dev=dev,
                    groupsize=args.group_size,
                    symmetric=not args.asym,
                    max_iter_num=args.max_iter_num,
                    inner_iters_for_round=args.inner_iters_for_round,
                    iters_before_round=args.iters_before_round,
                    lr=args.lr,
                    actorder=args.act_order,
                    round_fn=args.round_fn,
                )
                t2 = time.time()
                print(
                    f"time cost {t2 - t1}, model.decoder.layers.{i + shift}.{name}.weight, loss is {loss.mean().item()}")
                print()
                scale_list = [k.cpu() for k in [scale_out, zero_out]]
                quantizers[f"{i + shift}.{name}.weight"] = {
                    "scales": scale_list, "weights": w_int.cpu(), "loss": loss.cpu()}
                moq[name].free()
                moq[name].quantizer.free()
                del moq[name], scale_out, zero_out, w_int
        outs = []
        if args.blockwise_minimize_lr > 0:
            t1 = time.time()
            minimize_block(args, quantizers, layer, inps, dev, i + shift, masks)
            shutil.rmtree("./tmp_blockwise")
            print("time cost for block minimization:", time.time() - t1)

        layer = layer.to(dev)
        for b in inps:
            b = to_device(b, dev)
            outs.append(to_device(layer(*(b[0]), **b[1]), "cpu"))

        layers[i] = layer.cpu()
        del layer
        del moq
        torch.cuda.empty_cache()

        for j in range(len(outs)):
            inps[j][0][0] = outs[j][0]
        del outs
        print(f"quant layer {i} done! time cost {time.time() - t_layer0}")
        print()
    del inps
    model.config.use_cache = use_cache
    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
                       :, (i * model.seqlen):((i + 1) * model.seqlen)
                       ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    model.config.use_cache = use_cache
    return ppl.item(), (torch.stack(nlls).sum() / (nsamples * model.seqlen)).item()



def make_qw2_linear(old_linear: torch.nn.Linear, state_dicts, group_size, name):
    in_features = old_linear.in_features
    out_features = old_linear.out_features
    bias = old_linear.bias
    new_qw2_linear = LinearW2A16(in_features, out_features, bias, group_size)
    ## turn on and use fake_quant model to verify compute result
    #new_qw2_linear = LinearA16(in_features, out_features, bias, group_size)
    
    #add weight/bias/scale/zp from state_dict
    weight = state_dicts[f"{name}.weight"]
    if f"{name}.weight_qscale" in state_dicts:
        scale = state_dicts[f"{name}.weight_qscale"]
        zp = state_dicts[f"{name}.weight_qzero"]
        new_qw2_linear.scale = scale.cuda().half().t().contiguous()
        new_qw2_linear.zp = zp.cuda().half().t().contiguous()

    new_qw2_linear.weight = weight.t().contiguous()

    return new_qw2_linear

def replace_llama_with_w2(llama_model, state_dicts, group_size):
    layers = llama_model.model.layers
    for i in range(len(layers)):
        q_proj_new_linear = make_qw2_linear(layers[i].self_attn.q_proj, state_dicts, group_size, f"model.layers.{i}.self_attn.q_proj")
        k_proj_new_linear = make_qw2_linear(layers[i].self_attn.k_proj, state_dicts, group_size, f"model.layers.{i}.self_attn.k_proj")
        v_proj_new_linear = make_qw2_linear(layers[i].self_attn.v_proj, state_dicts, group_size, f"model.layers.{i}.self_attn.v_proj")
        o_proj_new_linear = make_qw2_linear(layers[i].self_attn.o_proj, state_dicts, group_size, f"model.layers.{i}.self_attn.o_proj")

        gate_proj_new_linear = make_qw2_linear(layers[i].mlp.gate_proj, state_dicts, group_size, f"model.layers.{i}.mlp.gate_proj")
        up_proj_new_linear = make_qw2_linear(layers[i].mlp.up_proj, state_dicts, group_size, f"model.layers.{i}.mlp.up_proj")
        down_proj_new_linear = make_qw2_linear(layers[i].mlp.down_proj, state_dicts, group_size, f"model.layers.{i}.mlp.down_proj")
        
        layers[i].self_attn.q_proj = q_proj_new_linear
        layers[i].self_attn.k_proj = k_proj_new_linear
        layers[i].self_attn.v_proj = v_proj_new_linear
        layers[i].self_attn.o_proj = o_proj_new_linear
        layers[i].mlp.gate_proj = gate_proj_new_linear
        layers[i].mlp.up_proj = up_proj_new_linear
        layers[i].mlp.down_proj = down_proj_new_linear

def llama_infer(llama_model, input_ids):
    pass


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--quant_pt', type=str,
        help='quant model to infer'
    )
    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.',
        default = 'c4'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--group-size', type=int, default=64,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order decoupleQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )
    parser.add_argument(
        '--quant-method', type=str, choices=['optq', 'moq', 'moq_sequential', ""], default="",
        help='the quant method'
    )
    parser.add_argument(
        '--loss-thr', type=float, default=0.02,
        help='The loss threshold to exit loop'
    )
    parser.add_argument(
        '--max-iter-num', type=int, default=3,
        help='The max iter num for the whole loop'
    )
    parser.add_argument(
        '--inner-iters-for-round', type=int, default=50,
        help='the number of iters for PGD when use first level approximation'
    )
    parser.add_argument(
        '--iters-before-round', type=int, default=0,
        help='the number of iters before entering PGD when use first level approximation'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='the learning rate for PGD'
    )
    parser.add_argument(
        '--round-fn', type=str, choices=["gptq", "train"], default="train",
        help='the quant method'
    )
    parser.add_argument(
        '--blockwise-minimize-lr', type=float, default=-1.0,
        help='the learning rate for block minimization'
    )
    parser.add_argument(
        '--blockwise-minimize-wd', type=float, default=1.0e-6,
        help='the weight decaying rate for block minimization'
    )
    parser.add_argument(
        '--blockwise-minimize-epoch', type=int, default=3,
        help='the number of epoch for training the float point part'
    )
    parser.add_argument(
        '--train-LN', action='store_true',
        help='Whether to train the parameters in norm'
    )
    parser.add_argument(
        '--train-bias', action='store_true',
        help='Whether to train the bias in linear layer'
    )
    parser.add_argument(
        '--inference', action = 'store_true',
        help = "inference trained model"
    )

    args = parser.parse_args()

    args.asym = not args.sym
    args.qbits = args.wbits

    model = get_llama(args.model)
    model.eval()

    if args.inference:
        from transformers import AutoTokenizer
        state_dict = torch.load(f"{args.quant_pt}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        replace_llama_with_w2(model, state_dict, args.group_size)

        prompts = ["who are you?"]
        input_token_ids = tokenizer(prompts)['input_ids']
        input_token_ids_tensor = torch.LongTensor(input_token_ids).cuda()
        
        model.cuda()

        import time

        with torch.cuda.amp.autocast(dtype=torch.float16):
            #do warmup
            model_output = model.generate(input_token_ids_tensor, max_length=40)
            model_output = model.generate(input_token_ids_tensor, max_length=40)

            t_start = time.time()
            model_output = model.generate(input_token_ids_tensor, max_length=40)
            t_end = time.time()

        out_text = tokenizer.batch_decode(model_output)
        infer_time = (t_end - t_start) * 1000
        print(f"out_text: {out_text}")
        print(f"inference speed: e2e {infer_time} ms, pertoken {infer_time / model_output.shape[-1]} ms")
    else:
        dataloader, testloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
        )

        dev = "cuda:0"
        rank = 0
        layers = model.model.layers
        dataloader = [b[0] for b in dataloader]
        tick = time.time()
        quantizers = quant_sequential(args, model, layers, dataloader, f"cuda:{rank}")
        print("The quantization duration is ", (time.time() - tick) / 3600)
        datasets = ['wikitext2', 'ptb', 'c4']
        if args.new_eval:
            datasets = ['wikitext2', 'ptb-new', 'c4-new']
        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print(dataset)
            ppl, logPPL = llama_eval(model, testloader, dev)
            print(f"=====The ppl of {dataset} is {ppl}, logPPL is {logPPL}")

        if args.save:
            llama_pack3(model, quantizers)
            torch.save(model.state_dict(), args.save)
