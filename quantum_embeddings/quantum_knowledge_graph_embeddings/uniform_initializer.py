from pykeen.nn.init import Initializer
import torch

class UniformInitializer(Initializer):
    def __init__(self, low: float = -0.01, high: float = 0.01):
        super().__init__()
        self.low = low
        self.high = high

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.uniform_(tensor, a=self.low, b=self.high)

if __name__ == '__main__':
    i = UniformInitializer(low = 2.0 , high = 3.0)
    t = torch.zeros(4)
    print(t)
    print(i(t))
    