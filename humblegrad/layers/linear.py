from humblegrad.core.tensor import Tensor
import numpy as np

class Linear:
    def __init__(self, in_features: int, out_features: int):
        self._in_features = in_features
        self._out_features = out_features
        
        limit = 1 / np.sqrt(in_features)
        # weight: (out_features, in_features)
        self.weight = Tensor(np.random.uniform(-limit, limit, (out_features, in_features)), requires_grad=True)
        
        # bias: (out_features,)
        self.bias = Tensor(np.random.uniform(-limit, limit, (out_features,)), requires_grad=True)
    
    def __call__(self, x: Tensor) -> Tensor:
        
        if x.data.ndim == 1:
            x = x.reshape(1, -1)
        
        # x: (batch_size, in_features)
        assert x.data.shape[1] == self._in_features, \
            f"Expected input features {self._in_features}, given {x.data.shape[1]}"
        
        # out: (batch_size, out_features)
        out = x @ self.weight.T + self.bias
        
        return out
        
        