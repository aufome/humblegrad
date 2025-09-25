from humblegrad.core.tensor import Tensor
from humblegrad.core.tensor import Node

def relu(tensor: Tensor) -> Tensor:
    data = tensor.data.copy()
    data[data < 0] = 0
    out = Tensor(data, requires_grad=tensor.requires_grad)

    if tensor.requires_grad:
        def backward_fn():
            grad = (tensor.data > 0).astype(tensor.data.dtype) * out.grad
            tensor.grad = (tensor.grad or 0) + grad

        out.node = Node("relu", (tensor,), out, backward_fn)

    return out
