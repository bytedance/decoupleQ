
from tokenize import group
import torch
from decoupleQ.linear_w2a16 import LinearW2A16
import time

def bench_w2a16_kernel(m, n, k, groupsize, with_bias):
    assert k % groupsize == 0, "invalid groupsize {groupsize} for k {k}"
    input = torch.randn(m, k, dtype = torch.float16).cuda()
    w2_weight = torch.randint(-8, 7, (k,  n)).to(torch.int8).cuda()
    group_cnt = k // groupsize
    w2_scale = torch.randn(group_cnt, n, dtype=torch.float16).cuda()
    w2_zp = torch.randn(group_cnt, n, dtype=torch.float16).cuda()
    bias = torch.randn(group_cnt, n, dtype=torch.float16).cuda()

    w2_scale_expand = w2_scale.reshape((group_cnt, 1, n)).repeat((1, groupsize, 1)).reshape((k, n)).contiguous()
    w2_zp_expand = w2_zp.reshape((group_cnt, 1, n)).repeat((1, groupsize, 1)).reshape((k, n)).contiguous()
    w16_weight = w2_weight.float().cuda() * w2_scale_expand + w2_zp_expand
    w16_weight = w16_weight.half()

    # run linear_w2a16
    w2a16_linear = LinearW2A16(k, n, with_bias, groupsize)
    w2a16_linear.weight = w2_weight
    if with_bias:
        w2a16_linear.bias = bias
    w2a16_linear.scale = w2_scale
    w2a16_linear.zp = w2_zp

    print(f"Gemm param: m={m}, n={n}, k={k}, with_bias: {with_bias}, groupsize: {groupsize}")

    #warmup
    w2a16_linear(input)

    t_start = time.time() 
    for i in range(10):
        w2a16_linear(input)
    t_end = time.time()
    print(f"w2a16 forward time: {(t_end - t_start) * 100} ms")

    
    a16_linear = torch.nn.Linear(k, n, with_bias).cuda()
    a16_linear.weight = torch.nn.Parameter(w16_weight.t().contiguous())
    if with_bias:
        a16_linear.bias = torch.nn.Parameter(bias)

    #warmup
    a16_linear(input)

    t_start = time.time() 
    for i in range(10):
        a16_linear(input)
    t_end = time.time()
    print(f"a16 forward time: {(t_end - t_start) * 100} ms")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--m', type=int,
        help='gemm m',
        default=4096
    )
    parser.add_argument(
        '--n', type=int,
        help='gemm n',
        default=4096
    )
    parser.add_argument(
        '--k', type=int,
        help='gemm k',
        default=4096
    )
    parser.add_argument(
        '--groupsize', type=int,
        help='quant groupsize',
        default=64
    )
    parser.add_argument(
        '--with_bias', action='store_true',
        help='gemm with bias or not'
    )

    args = parser.parse_args()

    bench_w2a16_kernel(args.m, args.n, args.k, args.groupsize, args.with_bias)







