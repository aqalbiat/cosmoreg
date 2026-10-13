# Define a compact MLP: 50 log-power features -> hidden layers -> 2 linear outputs.
# Outputs represent standardized Omega_m and sigma_8.

from torch import nn
from numbers import Integral


class SpectrumMLP(nn.Module):
    def __init__(self, hidden_dims=(32, 16)):
        super().__init__()

        if len(hidden_dims) != 2 or any(
            not isinstance(width, Integral) or width < 1 for width in hidden_dims
        ):
            raise ValueError("hidden_dims must contain two positive integer widths.")
        first_width, second_width = hidden_dims

        self.network = nn.Sequential(
            nn.Linear(50, first_width),
            nn.ReLU(),
            nn.Linear(first_width, second_width),
            nn.ReLU(),
            nn.Linear(second_width, 2)
        )

    def forward(self, X):
        return self.network(X)
