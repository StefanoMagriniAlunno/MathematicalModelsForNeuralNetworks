# una hopfield neaural network ha le seguenti caratteristiche:
# 1) ammette pattern di shape (P,) e ha come parametri una matrice PxP detta M
# 2) forward:
# 3) calcola A = M.T @ M
# 4) applica y = A @ softmax(x)

# 5) più canali -> convex merging finale
# 6) più pattern per classe -> max merging intermedia
# 7) comportamento variational gaussiano

import torch
from torch import nn
from torch.nn import functional


class HNN(nn.Module):

    n_neurons: int
    n_classes: int
    n_nodes: int
    n_channels: int
    n_patterns: int
    n_iterations: int
    _matrix_encoder: nn.Parameter
    _matrix_w: nn.Parameter
    _convex_w: nn.Parameter
    _eta_w: nn.Parameter
    _bias: nn.Parameter

    def __init__(
        self,
        n_neurons: int,
        n_classes: int,
        n_nodes: int = 1,
        n_channels: int = 1,
        n_iterations: int = 1,
        M_init_noise: float = 0.1,
        C_init_noise: float = 0.1,
        eta_init_noise: float = 0.1,
        bias_init_noise: float = 0.1,
    ):
        """Build a Hopfield Neural Network

        Args:
            n_neurons (int): Num of input neurons
            n_classes (int): Num of classes
            n_nodes (int): Num of nodes for each class
            n_channels (int): Number of channels (versions of the same Hopfield Neural Network)
            n_iterations (int): Number of iterations for the Hopfield Neural Network
            M_init_noise (float): Initialization noise for the matrix M
            C_init_noise (float): Initialization noise for the convex merging matrix
            eta_init_noise (float): Initialization noise for the eta parameter
            bias_init_noise (float): Initialization noise for the bias
        """
        super(HNN, self).__init__()

        assert n_neurons > 0, "n_neurons must be greater than 0"
        assert n_classes > 0, "n_classes must be greater than 0"
        assert n_nodes > 0, "n_nodes must be greater than 0"
        assert n_channels > 0, "n_channels must be greater than 0"
        assert n_iterations > 0, "n_iterations must be greater than 0"

        self.n_neurons = n_neurons
        self.n_classes = n_classes
        self.n_nodes = n_nodes
        self.n_channels = n_channels
        self.n_patterns = n_classes * n_nodes
        self.n_iterations = n_iterations

        self._matrix_encoder = nn.Parameter(
            torch.randn(self.n_channels, self.n_neurons, self.n_patterns),
            requires_grad=False,
        )

        self._matrix_w = nn.Parameter(
            torch.tanh(
                M_init_noise
                * torch.randn(self.n_channels, self.n_patterns, self.n_patterns)
            ),
            requires_grad=True,
        )

        self._convex_w = nn.Parameter(
            torch.tanh(C_init_noise * torch.randn(self.n_classes, self.n_channels)),
            requires_grad=True,
        )

        self._eta_w = nn.Parameter(
            eta_init_noise * torch.randn(self.n_channels) - 3, requires_grad=True
        )

        self._bias = nn.Parameter(
            bias_init_noise * torch.randn(self.n_classes), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, n_neurons)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_classes)
        """
        x = torch.einsum("bcn, cnp -> bcp", x, self._matrix_encoder)
        A = torch.einsum(
            "cip, cjp -> cij", self._matrix_w, self._matrix_w
        )  # i,j = pattern
        for _ in range(self.n_iterations):
            p = functional.softmax(x, dim=-1)
            # variational gaussian behaviour
            new_p = p.clone().detach()
            eps = torch.randn_like(p)  # (batch_size, n_channels, n_patterns, )
            p = p + eps * torch.sqrt(new_p * (1 - new_p)) * torch.exp(self._eta_w).view(
                1, -1, 1
            )
            x = torch.einsum("cpi, bci -> bcp", A, p)  # i = pattern

        # compute max merging
        x = torch.max(
            x.view(x.size(0), self.n_channels, self.n_classes, self.n_nodes), dim=-1
        )[0]

        # compute convex merging
        x = torch.einsum(
            "bck, kc -> bk", x, functional.softmax(self._convex_w, dim=-1)
        )  # k = class

        return x + self._bias


if __name__ == "__main__":
    hnn = HNN(16, 10, 2, 128, 2)
    x = torch.randn(100, 128, 16)
    y = hnn(x)
    print(y.shape)
