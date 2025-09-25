from humblegrad.core.tensor import Tensor
from humblegrad.core.tensor import Node
import numpy as np

def sigmoid(tensor: Tensor):
    data = 1 / (1 + np.exp(-tensor.data))
    out = Tensor(data, requires_grad=tensor.requires_grad)
    
    if out.requires_grad:
        if tensor.requires_grad:
            def backward_fn():
                tensor.grad = (tensor.grad or 0) + data * (1 - data) * out.grad
        
        out.node = Node("sigmoid", (tensor,), out, backward_fn)
        
    return out