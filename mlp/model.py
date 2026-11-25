# Define a compact MLP: log-power features -> hidden layers -> 2 linear outputs.
# Outputs represent standardized Omega_m and sigma_8.

from torch import nn
from numbers import Integral


class SpectrumMLP(nn.Module):
    def __init__(self, hidden_dims=(32, 16), *, input_dim=50):
        super().__init__()

        if not isinstance(input_dim, Integral) or input_dim < 1:
            raise ValueError("input_dim must be a positive integer.")
        if len(hidden_dims) != 2 or any(
            not isinstance(width, Integral) or width < 1 for width in hidden_dims
        ):
            raise ValueError("hidden_dims must contain two positive integer widths.")
        first_width, second_width = hidden_dims
        self.input_dim = int(input_dim)

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, first_width),
            nn.ReLU(),
            nn.Linear(first_width, second_width),
            nn.ReLU(),
            nn.Linear(second_width, 2)
        )

    def forward(self, X):
        return self.network(X)
