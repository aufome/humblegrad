
from humblegrad.core.tensor import Node, Tensor
import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_h = self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size

        # Xavier uniform initialization
        limit = 1 / np.sqrt(in_channels * self.kernel_h * self.kernel_w)
        self.weight = Tensor(
            np.random.uniform(-limit, limit,
                              (out_channels, in_channels, self.kernel_h, self.kernel_w)),
            requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        batch_size, in_channels_tensor, input_h, input_w = x.data.shape
        assert in_channels_tensor == self.in_channels, \
            f"Expected input with {self.in_channels} channels, got {in_channels_tensor}"

        output_h = input_h - self.kernel_h + 1
        output_w = input_w - self.kernel_w + 1

        output_data = np.zeros((batch_size, self.out_channels, output_h, output_w))

        # Forward convolution
        for b in range(batch_size):
            for out_ch in range(self.out_channels):
                for in_ch in range(self.in_channels):
                    for i in range(output_h):
                        for j in range(output_w):
                            window = x.data[b, in_ch, i:i+self.kernel_h, j:j+self.kernel_w]
                            output_data[b, out_ch, i, j] += np.sum(
                                window * self.weight.data[out_ch, in_ch]
                            )
                # Add bias
                output_data[b, out_ch] += self.bias.data[out_ch]

        out = Tensor(output_data, requires_grad=x.requires_grad)

        if out.requires_grad:
            def backward_fn():
                if x.requires_grad:
                    if x.grad is None:
                        x.grad = np.zeros_like(x.data)
                    for b in range(batch_size):
                        for out_ch in range(self.out_channels):
                            for in_ch in range(self.in_channels):
                                for i in range(output_h):
                                    for j in range(output_w):
                                        x.grad[b, in_ch, i:i+self.kernel_h, j:j+self.kernel_w] += \
                                            self.weight.data[out_ch, in_ch] * out.grad[b, out_ch, i, j]

                if self.weight.requires_grad:
                    if self.weight.grad is None:
                        self.weight.grad = np.zeros_like(self.weight.data)
                    for out_ch in range(self.out_channels):
                        for in_ch in range(self.in_channels):
                            for i in range(output_h):
                                for j in range(output_w):
                                    for b in range(batch_size):
                                        x_window = x.data[b, in_ch, i:i+self.kernel_h, j:j+self.kernel_w]
                                        self.weight.grad[out_ch, in_ch] += x_window * out.grad[b, out_ch, i, j]

                if self.bias.requires_grad:
                    if self.bias.grad is None:
                        self.bias.grad = np.zeros_like(self.bias.data)
                    for out_ch in range(self.out_channels):
                        self.bias.grad[out_ch] += np.sum(out.grad[:, out_ch])

            out.node = Node("conv2d", (x, self.weight, self.bias), out, backward_fn)

        return out
