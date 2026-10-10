# Define a compact MLP: 50 log-power features -> hidden layers -> 2 linear outputs.
# Outputs represent standardized Omega_m and sigma_8.

from torch import nn


class SpectrumMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, X):
        return self.network(X)
