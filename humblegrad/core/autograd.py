from .tensor import Tensor

class AutogradEngine:
    def __init__(self, tensor: Tensor):
        self._tensor = tensor
        
    def backward(self):
        for tensor in self.topological_sort():
            if tensor.node is not None:
                tensor.node.backward_fn()
    
    def topological_sort(self):
        seen = set()
        topological_order = []
        
        def traverse(tensor):
            if tensor not in seen:
                seen.add(tensor)
            if tensor.node is not None:
                for child in tensor.node.inputs:
                    traverse(child)
            topological_order.append(tensor)
            
        traverse(self._tensor)
        
        return reversed(topological_order)