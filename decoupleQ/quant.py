"""
Copyright (2024) Bytedance Ltd. and/or its affiliates
"""
import torch
import torch.nn as nn
from decoupleQ.moq_quant import Quantizer, repeat_interleave, opt_intW3
import torch.nn.functional as F


def find_layers(module, layers=(nn.Conv2d, nn.Linear), name=''):
    if isinstance(module, layers) or module.__class__.__name__ in ("Conv1D",):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        obj = obj.to(device)
        return obj
    elif isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            new_obj[k] = to_device(v, device)
        return new_obj
    elif isinstance(obj, (list, tuple)):
        new_obj = []
        for v in obj:
            new_obj.append(to_device(v, device))
        if isinstance(obj, tuple):
            new_obj = tuple(new_obj)
        return new_obj
    elif isinstance(obj, nn.Module):
        obj = obj.to(device)
        return obj
    return obj


def fp16tofloat(obj):
    if isinstance(obj, torch.Tensor) and obj.dtype in (torch.bfloat16, torch.float16):
        obj = obj.float()
        return obj
    elif isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            new_obj[k] = fp16tofloat(v)
        return new_obj
    elif isinstance(obj, (list, tuple)):
        new_obj = []
        for v in obj:
            new_obj.append(fp16tofloat(v))
        if isinstance(obj, tuple):
            new_obj = tuple(new_obj)
        return new_obj
    elif isinstance(obj, nn.Module):
        for n in list(obj.parameters()) + list(obj.buffers()):
            if n.dtype in (torch.bfloat16, torch.float16):
                n.data = n.data.float()
        return obj
    return obj


class decoupleQ(object):
    def __init__(self, layer, name=''):
        self.layer = layer
        W = layer.weight.data
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        elif self.layer.__class__.__name__ in ("Conv1D",) and W.ndim == 2:
            W = W.t()
        elif isinstance(self.layer, nn.Linear):
            pass
        else:
            raise NotImplementedError("not support yet")
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        # self.H = torch.zeros((self.columns, self.columns), dtype=torch.float, device=W.device)
        self.H = 0
        self.nsamples = 0

    def add_batch(self, inp, out, mask):
        if inp.isnan().any():
            print(f"catch a NAN!!!!!!")
            return
        mask = mask.to(inp.dtype) if mask is not None else None
        if isinstance(self.layer, nn.Linear) or self.layer.__class__.__name__ in ("Conv1D",):
            inp = inp.reshape((-1, inp.shape[-1]))  # [batch, dim]
            if mask is not None:
                mask = mask.reshape((-1, 1))  # [batch, dim]
            inp = inp * mask if mask is not None else inp
            inp = inp.t()  # [dim, batch]

        elif isinstance(self.layer, nn.Conv2d):  # [batch, channel, hight, width]
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        else:
            raise NotImplementedError("not support yet")
        # tmp = inp.shape[-1]
        tmp = mask.sum().double() if mask is not None else inp.shape[-1]
        inp = inp.double()
        h = inp.matmul(inp.t())
        self.H = (self.H * self.nsamples + h) / (self.nsamples + tmp)
        self.nsamples += tmp

    def startquant(self, groupsize, symmetric, max_iter_num,
                   inner_iters_for_round, iters_before_round, dev, lr, actorder=True,
                   round_fn="gptq", perdamp=0.01):
        W = self.layer.weight.data.clone().detach()  # [out_channel, in_channel, kernel_size, kernel_size]
        W = W.to(dev).float()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        elif self.layer.__class__.__name__ in ("Conv1D", ) and W.ndim == 2:
            W = W.t()
        elif isinstance(self.layer, nn.Linear):
            pass
        else:
            raise NotImplementedError

        trt = f"The number of samples for GPTQ must be larger than the dimension of Hession matrix;" \
              f"Otherwise, H must be a singular matrix and cannot be inverted. nsample {self.nsamples}, columns {self.columns}"
        # assert self.nsamples > self.columns, trt
        print(trt)
        H = self.H.to(device=dev, dtype=W.dtype)
        del self.H
        dp = torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[0], device=H.device)
        H[diag, diag] += perdamp * dp

        H00 = H
        W00 = W
        # ===================================================================================================
        iters_for_scale = self.quantizer.iters_for_scale
        self.quantizer.iters_for_scale = 0
        scale, zero, scale_out, zero_out, err = self.quantizer.find_params(W, groupsize=groupsize, H=H)
        print(f"get scale via minmax, the init loss is {err.mean().item()}")
        w_int = self.quantizer.get_fake_int_in(W00, scale, zero, groupsize=groupsize)
        del scale, zero
        if max_iter_num >= 1:
            # max_iter_num = 1 is GPTQ
            s0, z0 = repeat_interleave(W, groupsize, scale_out, zero_out)
            h00, w00 = H00.clone(), W00.clone()
            perm, invperm = None, None
            if actorder:
                perm = torch.argsort(torch.diag(H), descending=True)
                invperm = torch.argsort(perm)
                h00 = h00[perm][:, perm]
                s0, z0 = s0[:, perm], z0[:, perm]
                w00 = w00[:, perm]
            w_int0, _, err0 = opt_intW3(s0, z0, h00, symmetric, self.quantizer.min_bound,
                                        self.quantizer.max_bound, w00, round_fn="gptq")
            if actorder:
                w_int0 = w_int0[:, invperm]
            tmp = err0 < err
            w_int[tmp] = w_int0[tmp]
            err[tmp] = err0[tmp]
            rate = torch.sum(tmp) / tmp.shape[0]
            print(f"the success rate of gptq is {rate.item()}, the loss is {err.mean().item()}")

        torch.cuda.empty_cache()
        max_inner_iter = inner_iters_for_round
        iter_num = -1
        for iter_num in range(max_iter_num - 1):
            if iter_num == 0:
                self.quantizer.iters_for_scale = max(iters_for_scale, 2)
                _, _, scale_out0, zero_out0, err0 = self.quantizer.find_params(W, groupsize=groupsize, H=H)
            else:
                scale_out0, zero_out0 = scale_out, zero_out
            s0, z0 = repeat_interleave(W, groupsize, scale_out0, zero_out0)
            h00, w00 = H00.clone(), W00.clone()
            if actorder:
                h00 = h00[perm][:, perm]
                s0, z0 = s0[:, perm], z0[:, perm]
                w00 = w00[:, perm]
            w_int0, _, err0 = opt_intW3(s0, z0, h00, symmetric, self.quantizer.min_bound, self.quantizer.max_bound, w00,
                                        x_init=None, max_iter=iters_before_round, lr=lr, max_inner_iter=max_inner_iter,
                                        round_fn=round_fn)
            if actorder:
                w_int0 = w_int0[:, invperm]
            if (err0 < 0).any():
                eig = torch.linalg.eigh(H00)
                print("The eigenvalues is", eig.eigenvalues)
                print("The err is ", err0)
                print("The negative err is ", err0[err0 < 0])
                # from IPython import embed; embed(header="negative loss 0")
                # raise ValueError(f"Fatal error, the eigenvalues of hessian is {eig.eigenvalues}")
            tmp = err0 < err
            w_int[tmp] = w_int0[tmp]
            scale_out[tmp] = scale_out0[tmp]
            zero_out[tmp] = zero_out0[tmp]
            err[tmp] = err0[tmp]
            rate = torch.sum(tmp) / tmp.shape[0]
            print(f"Iter {iter_num}, the success rate of opt_intW is {rate.item()}, the loss is {err.mean().item()}")
            if rate < 1e-4:
                break
            del w_int0, err0, h00, w00, s0, z0, tmp, rate

            try:
                scale_out0, zero_out0, err0 = self.quantizer.get_scale_and_zero_out_group(
                    H=H00, groupsize=groupsize, x0=W00, x_int=w_int)
                if (err0 <= 0).any():
                    eig = torch.linalg.eigh(H00)
                    print(eig.eigenvalues)
                    raise ValueError(f"Fatal error, the eigenvalues os hessian is {eig.eigenvalues}")
            except torch.cuda.OutOfMemoryError:
                print("catch an OutOfMemoryError, the shape of the weight is ", W00.shape,
                      "we will spilt to get scale")
                num_part = 16
                num_channel = W00.shape[0]
                ps = num_channel // num_part
                if num_channel % num_part != 0:
                    break
                try:
                    scale_out0s, zero_out0s, err0s = [], [], []
                    # for oom. Cut along the out_channel dimension,
                    for k in range(num_part):
                        scale_out0, zero_out0, err0 = self.quantizer.get_scale_and_zero_out_group(
                            H=H00, groupsize=groupsize, x0=W00[k * ps:(k + 1) * ps],
                            x_int=w_int[k * ps:(k + 1) * ps])
                        scale_out0s.append(scale_out0)
                        zero_out0s.append(zero_out0)
                        err0s.append(err0)
                except torch.cuda.OutOfMemoryError:
                    print("catch an OutOfMemoryError again, the shape of the weight is ", W00.shape, "give up")
                    # If it still doesn’t work, give up.
                    torch.cuda.empty_cache()
                    break
                scale_out0, zero_out0, err0 = torch.cat(scale_out0s), torch.cat(zero_out0s), torch.cat(err0s)
            tmp = err0 < err
            err[tmp] = err0[tmp]
            scale_out[tmp] = scale_out0[tmp]
            zero_out[tmp] = zero_out0[tmp]
            rate = torch.sum(tmp) / tmp.shape[0]
            print(
                f"Iter {iter_num}, the success rate of analytic_scale is {rate.item()}, the loss is {err.mean().item()}")
            if rate < 1e-4:
                break
            del scale_out0, zero_out0, err0, tmp, rate

        print(f"after {iter_num} of pure_training, the loss is {err.mean().item()}")
        scale_out0, zero_out0 = repeat_interleave(W, groupsize, scale_out, zero_out)
        if symmetric:
            Q = w_int * scale_out0
        else:
            Q = w_int * scale_out0 + zero_out0
        loss = torch.matmul(torch.matmul((W00 - Q), H), (W00 - Q).t()).diag()
        print("finally the loss is ", loss.mean().item())
        if self.layer.__class__.__name__ in ("Conv1D",) and Q.ndim == 2:
            Q = Q.t()
            w_int = w_int.t()
            scale_out = scale_out.t()
            zero_out = zero_out.t()
        w_int = w_int.reshape(self.layer.weight.shape).to(torch.int8)
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        return scale_out, zero_out, w_int, err

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


def replace_forward(layers):
    # modify_forward(layers)
    origin_forward = {}
    for name, module in layers.named_modules():
        if isinstance(module, torch.nn.Linear) or module.__class__.__name__ in ("Conv1D", ):
            print(f"replace forward for {name}")
            origin_forward[name] = module.forward
            module.forward = linear_forward(module)
        elif isinstance(module, (torch.nn.Conv2d,)):
            print(f"replace forward for {name}")
            origin_forward[name] = module.forward
            module.forward = conv2d_forward(module)

    return origin_forward


def recover_forward(layers, origin_forward):
    for name, module in layers.named_modules():
        if name in origin_forward:
            print(f"recover forward for {name}")
            module.forward = origin_forward[name]


def linear_forward(self):
    def tmp(inputs, *args, **kwargs):
        shape = (len(inputs.shape) - 1) * [1] + [inputs.shape[-1]]
        if self.__class__.__name__ in ("Conv1D",):
            size_out = inputs.size()[:-1] + (self.nf,)
            dim = 0
            mdtype = "Conv1D"
        elif isinstance(self, (torch.nn.Linear,)):
            size_out = inputs.size()[:-1] + (self.weight.shape[0],)
            dim = 1
            mdtype = "Linear"
        else:
            raise NotImplementedError("Fatal Error")

        if hasattr(self, "scale"):
            if self.group_size == -1:
                self.group_size = self.weight.shape[dim]
            scale = torch.repeat_interleave(self.scale, repeats=self.group_size, dim=dim)
            zero = torch.repeat_interleave(self.zero, repeats=self.group_size, dim=dim)
            weight = self.weight * scale + zero
        else:
            weight = self.weight
        if mdtype == "Conv1D":
            out = torch.mm(inputs.view(-1, inputs.size(-1)), weight)
        else:
            out = F.linear(inputs, weight)
        if self.bias is not None:
            out = out + self.bias
        out = out.view(size_out)
        return out

    return tmp


def conv2d_forward(self):
    def tmp(inputs, *args, **kwargs):
        shape = [self.weight.shape[0]] + (len(self.weight.shape) - 1) * [1]
        if hasattr(self, "scale"):
            scale = torch.reshape(self.scale, shape=shape)
            zero = torch.reshape(self.zero, shape=shape)
            weight = self.weight * scale + zero
        else:
            weight = self.weight
        out = F.conv2d(inputs, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

    return tmp


@torch.enable_grad()
def minimize_block(args, quantizers, layer, inps, dev, layer_num, masks):
    layer = layer.to(dev)
    full = find_layers(layer)
    params = []
    original_dtype = {}
    for key in full:
        quantizer = quantizers[f"{layer_num}.{key}.weight"]
        weight = quantizer["weights"]
        scale_list = quantizer["scales"]
        original_dtype[key] = full[key].weight.dtype
        dtype = torch.float32
        factory_kwargs = {'device': dev, 'dtype': dtype}
        full[key].weight.data = weight.to(**factory_kwargs)
        full[key].weight.requires_grad_(False)
        scale = torch.nn.Parameter(scale_list[0].clone().to(**factory_kwargs), requires_grad=True)
        requires_grad = True if args.asym else False
        zero = torch.nn.Parameter(scale_list[1].clone().to(**factory_kwargs), requires_grad=requires_grad)
        full[key].register_parameter("scale", scale)
        full[key].register_parameter("zero", zero)
        full[key].group_size = args.group_size
        params.append(scale)
        if args.asym:
            params.append(zero)

    if args.train_LN:
        for k, m in layer.named_modules():
            if isinstance(m, (torch.nn.LayerNorm, torch.nn.BatchNorm2d)) or "Norm" in m.__class__.__name__:
                if hasattr(m, "weight"):
                    m.weight.requires_grad_(True)
                    params.append(m.weight)
                    print("add layer norm weight to train")
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad_(True)
                    params.append(m.bias)
                    print("add layer norm bias to train")

    if args.train_bias:
        for k, m in layer.named_modules():
            if isinstance(m, torch.nn.Linear) or m.__class__.__name__ in ("Conv1D", ):
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad_(True)
                    params.append(m.bias)
                    print("add linear bias to train")

    origin_forward = replace_forward(layer)
    lr = args.blockwise_minimize_lr
    opt = torch.optim.Adam(params, lr, eps=2.e-5, betas=(0.9, 0.99), weight_decay=args.blockwise_minimize_wd)
    print("--", opt.param_groups[0]["lr"])
    total_loss = 0.0
    for j in range(args.blockwise_minimize_epoch):
        for idx, b in enumerate(inps):
            b = fp16tofloat(to_device(b, dev))
            label = torch.load(f"./tmp_blockwise/out_{idx}.pth")
            mask = masks[idx]
            out = layer(*(b[0]), **(b[1]))

            res = (out[0] - to_device(label["out"][0], dev))
            if mask is not None:
                res = res * mask.float().unsqueeze(-1)
                loss = torch.sum(res * res) / (mask.float().sum() * res.shape[-1])
            else:
                loss = torch.mean(res * res)
            opt.zero_grad()
            loss.backward()
            total_loss += loss.item()
            opt.step()
        print(f"the avg loss for training scale zero is {total_loss / len(inps)}")
        total_loss = 0.0

    for key in full:
        scale, zero = full[key].scale, full[key].zero
        quantizers[f"{layer_num}.{key}.weight"]["scales"] = [scale.cpu(), zero.cpu()]
        if full[key].__class__.__name__ in ("Conv1D",):
            dim = 0
        elif isinstance(full[key], torch.nn.Linear):
            dim = 1
        else:
            raise NotImplementedError
        groupsize = args.group_size
        if groupsize == -1:
            groupsize = full[key].weight.data.shape[dim]
        scale = torch.repeat_interleave(scale, repeats=groupsize, dim=dim)
        zero = torch.repeat_interleave(zero, repeats=groupsize, dim=dim)
        full[key].weight.data = full[key].weight.data * scale + zero
        full[key].weight.data = full[key].weight.data.to(original_dtype[key])
        del full[key].scale, full[key].zero, full[key].group_size

    recover_forward(layer, origin_forward)
    print()
