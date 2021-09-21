from typing import List, Optional, Union
import torch


class Linear(torch.nn.Linear):
    """
    Wraps torch.nn.Linear for convenience.

    Arguments:
      in_size (int): Number of input neurons
      out_size (int): Number of output neurons
      weights (Optional[Union[torch.Tensor, List[float]]]):
          Optional weights. Defaults to kaiming uniform initiatization
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        weights: Optional[Union[torch.Tensor, List[float]]] = None,
    ):
        super().__init__(in_size, out_size, bias=False)
        if weights is not None:
            weight_tensor = torch.tensor(weights, dtype=torch.float32)
            assert weight_tensor.shape == (
                out_size,
                in_size,
            ), f"Expected weights of shape ({out_size}, {in_size}), but got {weight_tensor.shape}"
            self.weight = torch.nn.Parameter(weight_tensor)
