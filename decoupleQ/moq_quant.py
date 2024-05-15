"""
Copyright (2024) Bytedance Ltd. and/or its affiliates
"""
import torch.nn as nn
import torch
import math


def repeat_interleave(x, groupsize, scale, zero):
    groupsize = x.shape[1] if groupsize == -1 else groupsize
    if scale is not None:
        scale = torch.repeat_interleave(scale, repeats=groupsize, dim=1)
        shape1 = [scale.shape[0], scale.shape[1]] + [1] * (len(x.shape) - 2)
        scale = scale.reshape(shape1)
    if zero is not None:
        zero = torch.repeat_interleave(zero, repeats=groupsize, dim=1)
        shape1 = [zero.shape[0], zero.shape[1]] + [1] * (len(x.shape) - 2)
        zero = zero.reshape(shape1)
    return scale, zero


def cal_quant_error(x, q, h):
    assert x.ndim == 2
    assert q.ndim == 2
    assert h.ndim == 2
    err = torch.matmul(torch.matmul(q-x, h), (q-x).t()).diag()  # This line is much faster than the three lines below
    # r = q - x
    # err = torch.matmul(torch.matmul(r.unsqueeze(1), h.unsqueeze(0)), r.unsqueeze(-1))
    # err = err.squeeze(-1).squeeze(-1)
    return err  # [num_channel]


def my_round(x):
    # For customization, it’s better to add +0.5 instead of rounding towards an even number.
    # Ensure that the various methods that are mathematically consistent end up being behaviorally consistent.
    y = torch.round(x)
    idx = (y - x).abs() == 0.5
    if idx.any():
        x = x.clone()
        x[idx] += 0.5
        return torch.round(x)
    else:
        return y


class MyAdam(object):
    def __init__(self, param: torch.Tensor, lr: float) -> None:
        self.param = param
        self.lr = lr
        self.t = 0
        self.exp_avg = torch.zeros_like(param, memory_format=torch.preserve_format)
        self.exp_avg_sq = torch.zeros_like(param, memory_format=torch.preserve_format)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1.0e-5

    def step(self, grad):
        self.t += 1
        self.exp_avg.lerp_(grad, 1 - self.beta1)
        self.exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t
        step_size = self.lr / bias_correction1
        bias_correction2_sqrt = math.sqrt(bias_correction2)
        denom = (self.exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(self.eps)
        self.param.addcdiv_(self.exp_avg, denom, value=-step_size)


@torch.no_grad()
def coord_descent1(x, H00, left, right, start=0):
    """Using coordinate descent, solve x'Hx, and need to restrict x to be between [left, right].
    where H is a positive definite symmetric matrix .
    This version, after the gradient descent converges, converges further.
    It's just too slow. In the case where the diagonals of H00 are already in descending order,
    the convergence will be better.
    x: [num_channel, dim]
    H00: [dim, dim]
    left: [num_channel, dim]
    right: [num_channel, dim]
    """
    grad = torch.matmul(x, H00)  # coordinate descent，[num_channel, dim]
    for i in range(start, x.shape[1]):
        d = H00[i, i]
        other = grad - torch.outer(x[:, i], H00[i, :])
        x[:, i] = (-other[:, i] / d).clamp(min=left[:, i], max=right[:, i])
        grad = other + torch.outer(x[:, i], H00[i, :])
    return x


@torch.no_grad()
def opt_intW3(scale_out, zero_out, H00, sym, min_bound, max_bound, x0, x_init=None,
              max_iter=0, lr=1.e-3, max_inner_iter=0, round_fn="gptq"):
    l = min_bound * scale_out + zero_out
    r = max_bound * scale_out + zero_out
    left = torch.minimum(l, r)
    right = torch.maximum(l, r)
    w = x0.clone() if x_init is None else x_init.clone()
    opt = MyAdam(w, lr)
    max_iter = 0 if round_fn == "gptq" else max_iter
    for _ in range(max_iter):
        # Finding the gradient directly by yourself will be much faster than forward/backward.
        opt.step(torch.matmul(w - x0, H00))
        w.data.clamp_(min=left, max=right)
    # loss = torch.matmul(torch.matmul(w-x0, H00), (w-x0).t()).diag().mean().item()
    # print(f"rank-{rank}, after adam, the loss is {loss}")
    assert round_fn in ("gptq", "train")
    round_Fn = round_gptq if round_fn == "gptq" else round_train

    f1, w = round_Fn(w, scale_out, zero_out, sym, min_bound, max_bound, x0=x0, H00=H00,
                     left=left, right=right, max_iter=max_inner_iter, opt=opt)
    # loss = torch.matmul(torch.matmul((f1-x0), H00), (f1-x0).t()).diag()
    loss = cal_quant_error(f1, x0, H00)
    w_int = my_round(safe_div(f1 - zero_out, scale_out)).clamp(min=min_bound, max=max_bound)
    return w_int, w, loss


def round_gptq(W, scale_out, zero_out, sym, min_bound, max_bound, H00=None, **kwargs):
    newW = W.clone()
    Q = W.clone()
    mean_diag = torch.mean(torch.diag(H00)).double()
    diag = torch.arange(W.shape[1], device=W.device)
    for t in [0.01, 0.1, 1.0, 10.0, 100.0]:
        H = H00.clone().double()
        damp = t * mean_diag
        H[diag, diag] += damp
        H = torch.cholesky_inverse(torch.linalg.cholesky(H))
        H = torch.linalg.cholesky(H, upper=True)
        if not H.isnan().any():
            break
    if H.isnan().any():
        from IPython import embed;
        embed(header="nan appears!")
    H = H.to(W.dtype)
    Hinv = H
    for i in range(W.shape[1]):
        w = W[:, i]
        d = Hinv[i, i]
        newW[:, i] = w
        q = my_round(safe_div(w - zero_out[:, i], scale_out[:, i])).clamp(min=min_bound, max=max_bound)
        q = q * scale_out[:, i] + zero_out[:, i]
        q = q.flatten()
        Q[:, i] = q
        err1 = (w - q) / d
        W[:, i:] -= err1.unsqueeze(1).matmul(Hinv[i, i:].unsqueeze(0))
    return Q, newW


def round_train(w, scale_out, zero_out, sym, min_bound, max_bound, H00=None, x0=None, left=None,
                right=None, lr=0.001, max_iter=100, opt=None, **kwargs):
    exp_avg, exp_avg_sq = opt.exp_avg, opt.exp_avg_sq
    beta1, beta2 = opt.beta1, opt.beta2
    eps = 1.0e-4
    step = opt.t
    newW = w.clone()
    for i in range(0, w.shape[1] - 1):
        newW[:, i] = w[:, i]
        q = my_round(safe_div(w[:, i] - zero_out[:, i], scale_out[:, i])).clamp(min=min_bound, max=max_bound)
        q = q * scale_out[:, i] + zero_out[:, i]
        w[:, i] = q.flatten()
        exp_avg, exp_avg_sq = exp_avg[:, 1:], exp_avg_sq[:, 1:]
        l, r = left[:, (i + 1):], right[:, (i + 1):]
        h = H00[:, (i + 1):]
        for _ in range(max_iter):
            step += 1
            grad = torch.matmul(w, h) if x0 is None else torch.matmul(w - x0, h)
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            bias_correction2_sqrt = math.sqrt(bias_correction2)
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            w[:, (i + 1):].addcdiv_(exp_avg, denom, value=-step_size)
            w[:, (i + 1):].clamp_(min=l, max=r)
    newW[:, -1] = w[:, -1]
    w = my_round(safe_div(w - zero_out, scale_out)).clamp(min=min_bound, max=max_bound) * scale_out + zero_out
    return w, newW


def safe_div(x, y: torch.Tensor):
    # compute x/y element-wise safely
    sign = torch.sign(y)
    sign[sign == 0] = 1
    y = y.abs().clamp(min=1.0e-15) * sign  # assume torch.float32
    return x / y


def quantize(x, scale, zero, min_bound, max_bound, scale_out=None, zero_out=None, return_int=False, zero_no_shift=False):
    if zero_no_shift:
        q = torch.clamp(my_round(x / scale) + zero, min=min_bound, max=max_bound)
    else:
        q = torch.clamp(my_round((x - zero) / scale), min=min_bound, max=max_bound)
    if return_int:
        return q
    if scale_out is not None:
        return q * scale_out + zero_out
    if zero_no_shift:
        return scale * (q - zero)
    return scale * q + zero


def cal_quant_err_scale_zero(x, scale, zero, scale_out, zero_out, sym, min_bound, max_bound, groupsize, h=None, x0=None,
                             zero_no_shift=False):
    scale, zero = repeat_interleave(x, groupsize, scale, zero)
    scale_out, zero_out = repeat_interleave(x, groupsize, scale_out, zero_out)
    x0 = x.clone() if x0 is None else x0
    q = quantize(x, scale, zero, min_bound=min_bound, max_bound=max_bound, scale_out=scale_out, zero_out=zero_out,
                 zero_no_shift=zero_no_shift)
    err = cal_quant_error(x0, q, h)
    return err


def get_linear_eq(a, b, c, d, e, f):
    """
    Solve linear equations:
    ax+by=c
    dx+ey=f
    """
    deno = a * e - b * d
    sign = torch.sign(deno)
    sign[sign == 0] = 1.0
    bu = sign * 1e-5
    singular = deno.abs() < 1e-5
    deno[singular] = bu[singular]
    x = (c * e - b * f) / deno
    y = (a * f - c * d) / deno
    return x, y


class Quantizer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass

    def configure(self, bits, perchannel=False, sym=True, grid=200, maxshrink=0.8, thr=0.01, iters_for_scale=4,
                  zero_no_shift=False):
        # self.min_bound = 0
        # self.max_bound = 2 ** bits - 1
        self.max_bound = 2 ** (bits - 1) - 1
        self.min_bound = -self.max_bound - 1

        self.perchannel = perchannel
        self.sym = sym
        self.grid = grid
        self.maxshrink = maxshrink
        self.thr = thr
        self.iters_for_scale = iters_for_scale
        self.zero_no_shift = zero_no_shift

    def find_params(self, x, groupsize=-1, H=None, best=None, x0=None):
        xmin, xmax = self.get_minmax(x, groupsize)
        scale, zero = self.get_scale_zero_by_minmax(xmax, xmin)  # [num_channel, num_group]
        if self.zero_no_shift:
            scale_out, zero_out = scale.clone(), -scale * zero
        else:
            scale_out, zero_out = scale.clone(), zero.clone()
        best = cal_quant_err_scale_zero(x, scale, zero, scale_out, zero_out, self.sym, self.min_bound, self.max_bound,
                                        groupsize, H, x0=x0, zero_no_shift=self.zero_no_shift)

        if self.iters_for_scale == 0:
            return scale, zero, scale_out, zero_out, best

        scale, zero, best, _ = self.get_scale_and_zero(xmin, xmax, x, H, best=best, scale=scale, zero=zero,
                                                       groupsize=groupsize, x0=x0)
        if self.zero_no_shift:
            scale_out, zero_out = scale.clone(), -scale * zero
        else:
            scale_out, zero_out = scale.clone(), zero.clone()

        if self.iters_for_scale == 1:
            return scale, zero, scale_out, zero_out, best
        try:
            scale_out, zero_out, best = self.get_scale_and_zero_out_group(x, scale, zero, H, groupsize, x0=x0)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        return scale, zero, scale_out, zero_out, best

    def get_scale_and_zero_out_group(self, x=None, scale=None, zero=None, H=None, groupsize=-1, x0=None, x_int=None,
                                     double_precision=True):
        x0 = x if x0 is None else x0
        if groupsize == -1 or groupsize == x0.shape[1]:
            return self.get_scale_and_zero_out(x, scale, zero, H, x0=x0, x_int=x_int)
        if x_int is None:
            x_int = self.get_fake_int_in(x, scale, zero, groupsize=groupsize)
        assert not self.sym  # There are groups, no symmetric quantization by default.

        def matmul(i, j, k):
            return torch.matmul(torch.matmul(i, j.unsqueeze(0)), k)

        num_channel = x_int.shape[0]
        dim = x_int.shape[1]
        num_group = dim // groupsize
        a = x_int

        # the code below is a bit ugly， to save GPU memory
        torch.cuda.empty_cache()
        A = torch.zeros((num_channel, num_group, dim), dtype=x_int.dtype, device=x_int.device)
        for j in range(num_group):
            A[:, j, j * groupsize: (j + 1) * groupsize] = a[:, j * groupsize: (j + 1) * groupsize]
        AH = torch.matmul(A, H.unsqueeze(0))
        A = torch.transpose(A, 1, 2)
        torch.cuda.empty_cache()
        P11 = torch.matmul(AH, A)
        del A
        torch.cuda.empty_cache()
        I = torch.zeros((num_channel, num_group, dim), dtype=x_int.dtype, device=x_int.device)
        for j in range(num_group):
            I[:, j, j * groupsize: (j + 1) * groupsize] = 1.0
        P12 = torch.matmul(AH, torch.transpose(I, 1, 2))
        del AH
        P22 = matmul(I, H, torch.transpose(I, 1, 2))
        del I
        torch.cuda.empty_cache()
        P21 = torch.transpose(P12, 1, 2)
        P = torch.cat([torch.cat((P11, P12), dim=2), torch.cat((P21, P22), dim=2)], dim=1).to(x_int.dtype)
        del P11, P12, P21, P22

        left = torch.matmul(x0, H)  # [channel, dim]
        left = torch.reshape(left, (num_channel, num_group, groupsize))
        up = torch.mul(left, a.reshape(num_channel, num_group, groupsize)).sum(2)  # [channel, num_group]
        down = left.sum(2)  # [channel, num_group]
        y = torch.cat([up, down], dim=1)  # [channel, 2* num_group]
        P = (P + P.transpose(1, 2)) / 2.0  # P must be a symmetric
        if double_precision:
            P, y = P.double(), y.double()
        try:
            scale_zero = torch.linalg.solve(P, y)
        except:
            diag = torch.arange(P.shape[-1], device=P.device)
            dP = P[:, diag, diag]  # [4608, 24]
            damp = 0.01 * torch.mean(dP, dim=1, keepdim=True)  # [4608,1]
            P[:, diag, diag] += damp
            scale_zero = torch.linalg.solve(P, y)
        scale_zero = scale_zero.float()
        scale_out, zero_out = torch.split(scale_zero, num_group, dim=1)
        if self.zero_no_shift:
            zero_out = my_round(zero_out / scale_out) * scale_out
        scale_out0, zero_out0 = repeat_interleave(x_int, groupsize, scale_out, zero_out)
        x = x_int * scale_out0 if self.sym else x_int * scale_out0 + zero_out0
        err = cal_quant_error(x0, x, H)
        return scale_out, zero_out, err

    def get_scale_and_zero(self, xmin, xmax, x, h, scale_out=None, zero_out=None, best=None, scale=None, zero=None,
                           groupsize=-1, x0=None):
        x0 = x if x0 is None else x0
        if best is None:
            best = torch.full([x.shape[0]], float("inf"), device=x.device)
        else:
            best = best.clone()
        if scale is not None:
            scale, zero = scale.clone(), zero.clone()
            eary_exit = False
        else:
            scale, zero = self.get_scale_zero_by_minmax(xmax, xmin)
            eary_exit = True
        bestp = torch.ones(x.shape[0], dtype=xmin.dtype, device=xmin.device)
        for i in range(int(self.maxshrink * self.grid)):  # 0.8*200
            p = 1 - i / self.grid
            xmin1 = p * xmin
            xmax1 = p * xmax
            scale1, zero1 = self.get_scale_zero_by_minmax(xmax1, xmin1)
            err = cal_quant_err_scale_zero(x, scale1, zero1, scale_out, zero_out, self.sym, self.min_bound,
                                           self.max_bound, groupsize, h, x0=x0, zero_no_shift=self.zero_no_shift)
            tmp = err < best
            if torch.any(tmp):
                best[tmp] = err[tmp]
                scale[tmp] = scale1[tmp]
                zero[tmp] = zero1[tmp]
                bestp[tmp] = p
            elif eary_exit:
                return scale, zero, best, bestp
        return scale, zero, best, bestp

    def get_minmax(self, x, groupsize):
        if self.perchannel:
            x = x.flatten(1)  # [num_channel, dim]
        else:
            x = x.flatten().unsqueeze(0)
        if groupsize == -1 or groupsize == x.shape[1]:
            num_group = 1
            groupsize = x.shape[1]
        else:
            assert x.shape[1] % groupsize == 0
            num_group = x.shape[1] // groupsize
        x = torch.reshape(x, (x.shape[0], num_group, groupsize))
        tmp = torch.zeros(x.shape[:2], device=x.device)
        xmin = torch.minimum(x.min(2)[0], tmp)  # [num_channel, num_group]
        xmax = torch.maximum(x.max(2)[0], tmp)
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1
        return xmin, xmax

    def quant_via_minmax(self, x, groupsize, h, bestp=None):
        xmin, xmax = self.get_minmax(x, groupsize)
        if bestp is None:
            scale, zero, _, bestp = self.get_scale_and_zero(xmin, xmax, x, h, groupsize=groupsize)
        else:
            shape = [xmax.shape[0]] + [1] * (xmax.dim() - 1)
            bestp1 = torch.reshape(bestp, shape)
            scale, zero = self.get_scale_zero_by_minmax(xmax * bestp1, xmin * bestp1)  # [num_channel, num_group]
        scale, zero = repeat_interleave(x, groupsize, scale, zero)
        x = quantize(x, scale, zero, self.min_bound, self.max_bound, zero_no_shift=self.zero_no_shift)
        return x, bestp


    def get_fake_int_in(self, x, scale, zero, groupsize=-1, zero_no_shift=None):
        scale, zero = repeat_interleave(x, groupsize, scale, zero)
        if zero_no_shift is None:
            zero_no_shift = self.zero_no_shift
        return quantize(x, scale, zero, self.min_bound, self.max_bound, None, None, return_int=True,
                        zero_no_shift=zero_no_shift)

    def get_scale_and_zero_out(self, x, scale, zero, h, x0=None, x_int=None):
        x0 = x if x0 is None else x0
        x_int = self.get_fake_int_in(x, scale, zero) if x_int is None else x_int
        if self.sym:
            scale_out, err = self.get_scale_out_sym(h, x0, x_int, -1)
            # zero_out = torch.zeros_like(scale_out)
            zero_out = torch.full_like(scale_out, (self.max_bound + self.min_bound + 1) // 2)
            return scale_out, zero_out, err
        # a convex function, and theoretically it must reach the optimal solution.
        # x_int, x [out_channel, dim]
        a = x_int  # [out_channel, dim]
        x0 = x0  # [out_channel, dim]
        I = torch.ones_like(a)  # [out_channel, dim]
        h = (h + h.t()) / 2.0  # h must be symmetric [dim, dim]
        Ih = torch.matmul(I, h)  # [out_channel, dim]
        ah = torch.matmul(a, h)  # [out_channel, dim]
        p1 = torch.matmul(ah, a.t()).diag()  # [out_channel]
        p2 = torch.matmul(I, ah.t()).diag()  # [out_channel]
        p3 = torch.matmul(x0, ah.t()).diag()  # [out_channel]
        p4 = p2
        p5 = torch.matmul(Ih, I.t()).diag()
        p6 = torch.matmul(x0, Ih.t()).diag()
        scale_out, zero_out = get_linear_eq(p1, p2, p3, p4, p5, p6)
        if torch.any(torch.isnan(scale_out) | torch.isinf(scale_out)) or torch.any(
                torch.isnan(zero_out) | torch.isinf(zero_out)):
            raise ValueError("Fatal Error")
        scale_out, zero_out = scale_out.unsqueeze(1), zero_out.unsqueeze(1)
        if self.zero_no_shift:
            zero_out = my_round(zero_out / scale_out) * scale_out
        scale_out0, zero_out0 = repeat_interleave(x_int, -1, scale_out, zero_out)
        q = x_int * scale_out0 if self.sym else x_int * scale_out0 + zero_out0
        err = cal_quant_error(x0, q, h)
        return scale_out, zero_out, err

    def get_scale_out_sym(self, h, x0, x_int, groupsize=-1):
        assert self.sym
        h = (h + h.t()) / 2.0  # h must be symmetric [dim, dim]
        ha = torch.matmul(h, x_int.t())  # [dim, num_channel]
        wha = torch.matmul(x0, ha).diag()  # [num_channel]
        aha = torch.matmul(x_int, ha).diag()  # [num_channel]
        scale_out = safe_div(wha, aha).unsqueeze(1)
        scale_out0, _ = repeat_interleave(x_int, groupsize, scale_out, None)
        q = x_int * scale_out0
        err = cal_quant_error(x0, q, h)
        return scale_out, err

    def get_scale_zero_by_minmax(self, xmax, xmin):
        scale = (xmax - xmin) / (self.max_bound - self.min_bound)
        if self.sym:
            zero = torch.full_like(scale, (self.max_bound + self.min_bound + 1) // 2)
        elif self.zero_no_shift:
            zero = self.min_bound - my_round(xmin / scale)
        else:
            zero = xmin - scale * self.min_bound
        return scale, zero

    def free(self):
        pass
