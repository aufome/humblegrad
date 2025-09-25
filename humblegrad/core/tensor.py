import numpy as np

__all__ = ["Tensor", "Node"]

class Node:
    """Internal computation graph node."""
    def __init__(self, op, inputs, output, backward_fn):
        self.op = op
        self.inputs = inputs
        self.output = output
        self.backward_fn = backward_fn
    
    def __repr__(self):
        return f"Node({self.op}: {self.inputs})"


class Tensor:
    def __init__(self, data, requires_grad=False, dtype = "float32"):
        self.data = np.array(data, dtype=dtype)
        self._grad = None
        self.requires_grad = requires_grad
        self.node = None
        
    @property
    def grad(self):
        return self._grad
    
    @grad.setter
    def grad(self, value):
        if value is None:
            self._grad = np.zeros_like(self.data) if self.requires_grad else None
        else:
            self._grad = value
            
    @property
    def T(self):
        out = Tensor(self.data.T, requires_grad=self.requires_grad)
        
        if out.requires_grad:
            def backward_fn():
                self.grad = (self.grad or 0) + out.grad.T
        
            out.node = Node("T", (self,), out, backward_fn)
        
        return out

        
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, self.requires_grad)
        out = Tensor(self.data + other.data, requires_grad= self.requires_grad or other.requires_grad)
        
        if out.requires_grad:
            def backward_fn():
                if self.requires_grad:
                    self.grad = (self.grad or 0) + 1 * out.grad
                if other.requires_grad:
                    other.grad = (other.grad or 0) + 1 * out.grad
            
            out.node = Node("+", (self, other), out, backward_fn)
            
        return out
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, self.requires_grad)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad)
        
        if out.requires_grad:
            def backward_fn():
                if self.requires_grad:
                    self.grad = (self.grad or 0) + other.data * out.grad
                if other.requires_grad:
                    other.grad = (other.grad or 0) + self.data * out.grad
            
            out.node = Node("*", (self, other), out, backward_fn)
        
        return out
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "powers should be int or float"
        out = Tensor(self.data ** other, self.requires_grad)
        
        if out.requires_grad:
            def backward_fn():
                if self.requires_grad:
                    self.grad = (self.grad or 0) + other * self.data ** (other - 1) * out.grad
            
            out.node("^", (self,), out, backward_fn)
        
        return out
    
    def __truediv__(self, other):
        assert other != 0, "division by zero is not allowed"
        other = other if isinstance(other, Tensor) else Tensor(other, self.requires_grad)
        out = Tensor(self.data / other.data, self.requires_grad or other.requires_grad)
        
        if out.requires_grad:
            def backward_fn():
                if self.requires_grad:
                    self.grad = (self.grad or 0) + (1 / other.data) * out.grad
                if other.requires_grad:
                    other.grad = (other.grad or 0) + -1 * self.data / (other.data ** 2) * out.grad
            
            out.node = Node("/", (self, other), out, backward_fn)
        return out
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other.__truediv__(self)
    
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, self.requires_grad)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad)
        
        if out.requires_grad:
            def backward_fn():
                # (m, n) @ (n, o) = (m, o)
                if self.requires_grad:
                    self.grad = (self.grad or 0) + out.grad @ other.data.T
                if other.requires_grad:
                    other.grad = (other.grad or 0) + self.data.T @ out.grad
            
            out.node = Node("@", (self, other), out, backward_fn)
    
        return out

    def __repr__(self):
        data_str = np.array2string(
            self.data,
            separator=', ',
            prefix='tensor('
        )
        extras = []
        if self.requires_grad:
            extras.append("requires_grad=True")
        if str(self.data.dtype) != "float32":
            extras.append(f"dtype={self.data.dtype}")
        extras_str = "" if not extras else ", " + ", ".join(extras)
        return f"tensor({data_str}{extras_str})"
    
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad)
        if out.requires_grad:
            def backward_fn():
                if self.requires_grad:
                    self.grad = (self.grad or 0) + out.grad.reshape(self.data.shape)
            out.node = Node("reshape", (self,), out, backward_fn)
        return out

    
    def backward(self):
        from .autograd import AutogradEngine
        self.grad = np.ones_like(self.data)
        engine = AutogradEngine(self)
        engine.backward()


