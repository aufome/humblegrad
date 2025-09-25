from humblegrad.core.tensor import Tensor
from humblegrad.core.tensor import Node
import numpy as np

def tanh(tensor: Tensor) -> Tensor:
    data = np.tanh(tensor.data)
    out = Tensor(data, requires_grad=tensor.requires_grad)

    if tensor.requires_grad:
        def backward_fn():
            grad = (1 - data ** 2) * out.grad
            tensor.grad = (tensor.grad or 0) + grad

        out.node = Node("tanh", (tensor,), out, backward_fn)

    return out
