import numpy
import torch
from torch import nn
from torch.autograd import Function

class Contiguous(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x:torch.Tensor):
        return x.contiguous()

class DropITer(object):
    def __init__(self, strategy, gamma, d=None):
        self.gamma = gamma
        self.reserve = 1 - gamma
        if d is not None:
            self.d = d
            self.d_reserve = int((1 - gamma) * d)
        self.select = getattr(self, f"select_{strategy}")
        self.pad = getattr(self, f"pad_{strategy}")
        self.nsigma = torch.sqrt(torch.tensor(2)) * torch.erfinv(2 * torch.tensor(gamma) - 1)
        self.dnsigma = torch.sqrt(torch.tensor(2)) * torch.erfinv(torch.tensor(gamma))
    
    # --- RANDOM CHANNEL ---  
    def select_random_c(self, x: torch.Tensor):
        c = x.shape[-1]
        x = x.view(-1, c)
        idxs = numpy.random.choice(c, int(c * self.reserve))
        x = x[:,idxs]
        x.idxs = idxs
        return x

    def pad_random_c(self, x: torch.Tensor, ctx):
        _x = torch.zeros(ctx.input_shape, device=x.device)
        _x.view(-1,_x.shape[-1])[:,x.idxs] = x
        del x.idxs
        return _x
    # --- RANDOM CHANNEL ---  

    # --- MINK CHANNEL ---  
    def select_mink_c(self, x: torch.Tensor):
        c = x.shape[-1]
        x = x.view(-1, c)
        idxs = x.norm(p=2, dim=0).topk(int(c * self.reserve), sorted=False)[1]
        x = x[:,idxs]
        x.idxs = idxs
        return x

    def pad_mink_c(self, x: torch.Tensor, ctx):
        c = ctx.input_shape[-1]
        _x = torch.zeros(ctx.input_shape, device=x.device)
        _x.view(-1, c)[:,x.idxs] = x
        del x.idxs
        return _x
    # --- MINK CHANNEL ---    

    # --- RANDOM ---    
    def select_random(self, x: torch.Tensor):
        x = x.view(-1)
        idxs = torch.from_numpy(numpy.random.choice(len(x), int(len(x) * self.reserve))).to(x.device)
        x = x[idxs]
        x.idxs = idxs
        return x
    
    def pad_random(self, x, ctx):
        _x = torch.zeros(ctx.input_shape, device=x.device)
        _x.view(-1)[x.idxs] = x
        del x.idxs
        return _x
    # --- RANDOM ---  

    # --- MINK ---  
    def select_mink(self, x: torch.Tensor):
        x, idxs = x.view(-1).topk(int(x.numel() * self.reserve), sorted=False)
        x.idxs = idxs
        return x
    
    def pad_mink(self, x, ctx):
        _x = torch.zeros(ctx.input_shape, device=x.device)
        _x.view(-1).scatter_(0, x.idxs, x)
        del x.idxs
        return _x
    # --- MINK --- 

    # --- PARALLEL MINK ---  
    def select_parallel_mink(self, x: torch.Tensor):
        x, idxs = x.view(-1, self.d).topk(self.d_reserve, dim=1, sorted=False)
        x.idxs = idxs
        return x
        
    def pad_parallel_mink(self, x, ctx):
        _x = torch.zeros(ctx.input_shape, device=x.device)
        _x.view(x.shape[0], -1).scatter_(1, x.idxs, x)
        del x.idxs
        return _x
    # --- PARALLEL MINK ---  

    # --- NSIGMA ---  
    def select_nsigma(self, x: torch.Tensor):
        if hasattr(x, "idxs"):
            idxs = x.idxs
            x = x.view(-1)
        else:
            x = x.view(-1)
            mean, std = x.mean(), x.std(unbiased=False) 
            t1, t2 = mean - self.dnsigma * std, mean + self.dnsigma * std
            idxs = ((x < t1) | (x > t2)).nonzero(as_tuple=True)[0]
        x = x[idxs]
        x.idxs = idxs
        return x
        
    def pad_nsigma(self, x, ctx):
        _x = torch.zeros(ctx.input_shape, device=x.device, dtype=x.dtype)
        _x.view(-1).scatter_(0, x.idxs, x)
        del x.idxs
        return _x
    # --- NSIGMA ---  
    
    # --- GAMMABIT ---  
    def select_gammabit(self, x: torch.Tensor):
        if self.gamma == 8:
            return torch.quantize_per_tensor(x, 0.1, 10, torch.quint8)
        elif self.gamma == 16:
            return x.half()

    def pad_gammabit(self, x, ctx):
        if self.gamma == 8:
            return x.dequantize()
        elif self.gamma == 16:
            return x.to(torch.float32)
    # --- GAMMABIT ---  

class _DropITLinear(Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: nn.Parameter,
        bias: nn.Parameter or None,
        dropiter: DropITer,
    ):
        ctx.dropiter = dropiter
        ctx.input_shape = input.shape
        ctx.has_bias = bias is not None
        output = torch.functional.F.linear(input, weight, bias)
        ctx.save_for_backward(dropiter.select(input), weight)
        return output
    
    @staticmethod 
    def backward(ctx, grad_output: torch.Tensor):
        input, weight = ctx.saved_tensors
        grad_bias = torch.sum(grad_output, 
            list(range(grad_output.dim()-1))) if ctx.has_bias else None
        input = ctx.dropiter.pad(input, ctx)
        grad_input = grad_output.matmul(weight.to(grad_output.dtype))
        ic, oc = input.shape[-1], grad_output.shape[-1]
        grad_weight = grad_output.view(-1,oc).T.mm(input.view(-1,ic).to(grad_output.dtype))
        return grad_input, grad_weight, grad_bias, None

class DropITLinear(nn.Module):
    def __init__(self, linear: nn.Linear, dropiter: DropITer):
        super().__init__()
        self.linear = linear
        self.dropiter = dropiter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear = self.linear
        if self.training:
            return _DropITLinear.apply(
                x, linear.weight, linear.bias,
                self.dropiter
            )
        else:
            return linear(x)

Implemented = {
    'Linear': DropITLinear,
}

def to_dropit(model: nn.Module, dropiter: DropITer):
    for child_name, child in model.named_children():
        type_name = type(child).__name__
        if type_name in Implemented:
            setattr(model, child_name, Implemented[type_name](child, dropiter))
            print(f"{type(child).__name__} -> {Implemented[type_name].__name__}")
        else:
            to_dropit(child, dropiter)

if __name__ == "__main__":
    pass