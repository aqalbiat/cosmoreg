"""
4-step CNNs.

Narrow (16 filters):  Model1 (3x3), Model2 (5x5), Model3 (7x7)
Wide (128 filters):   Model4 (3x3), Model5 (5x5)

Model4 is the wide version of Model1, and Model5 the wide version of Model2.
"""
import torch
import torch.nn as nn


class CNN4Steps_Model1(nn.Module):
    """Narrow baseline: one 3x3 conv with 16 filters."""

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # Feature extraction
            nn.Conv2d(1, 16, kernel_size=3, padding=1),     # Step 1: Conv (256x256)
            nn.LeakyReLU(0.2),                              # Step 2: Activation
            nn.AdaptiveAvgPool2d((2, 2)),                   # Step 3: Pool (shrinks to 2x2)

            # Decision
            nn.Flatten(),                                   # 16 * 2 * 2 = 64 values
            nn.Linear(16 * 2 * 2, 2),                       # Step 4: Output (Omega_m, sigma_8)
        )

    def forward(self, x):
        return self.net(x)


class CNN4Steps_Model2(nn.Module):
    """Narrow: Model1 with a 5x5 kernel."""

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # Feature extraction
            nn.Conv2d(1, 16, kernel_size=5, padding=2),     # Step 1: Conv (256x256)
            nn.LeakyReLU(0.2),                              # Step 2: Activation
            nn.AdaptiveAvgPool2d((2, 2)),                   # Step 3: Pool (shrinks to 2x2)

            # Decision
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 2),                       # Step 4: Output
        )

    def forward(self, x):
        return self.net(x)


class CNN4Steps_Model3(nn.Module):
    """Narrow: Model1 with a 7x7 kernel."""

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # Feature extraction
            nn.Conv2d(1, 16, kernel_size=7, padding=3),     # Step 1: Conv (256x256)
            nn.LeakyReLU(0.2),                              # Step 2: Activation
            nn.AdaptiveAvgPool2d((2, 2)),                   # Step 3: Pool (shrinks to 2x2)

            # Decision
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 2),                       # Step 4: Output
        )

    def forward(self, x):
        return self.net(x)


class CNN4Steps_Model4(nn.Module):
    """Wide: Model1 with 128 filters."""

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # Feature extraction
            nn.Conv2d(1, 128, kernel_size=3, padding=1),    # Step 1: Conv (256x256)
            nn.LeakyReLU(0.2),                              # Step 2: Activation
            nn.AdaptiveAvgPool2d((2, 2)),                   # Step 3: Pool (shrinks to 2x2)

            # Decision
            nn.Flatten(),                                   # 128 * 2 * 2 = 512 values
            nn.Linear(128 * 2 * 2, 2),                      # Step 4: Output
        )

    def forward(self, x):
        return self.net(x)


class CNN4Steps_Model5(nn.Module):
    """Wide: Model2 (5x5 kernel) with 128 filters."""

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # Feature extraction
            nn.Conv2d(1, 128, kernel_size=5, padding=2),    # Step 1: Conv (256x256)
            nn.LeakyReLU(0.2),                              # Step 2: Activation
            nn.AdaptiveAvgPool2d((2, 2)),                   # Step 3: Pool (shrinks to 2x2)

            # Decision
            nn.Flatten(),                                   # 128 * 2 * 2 = 512 values
            nn.Linear(128 * 2 * 2, 2),                      # Step 4: Output
        )

    def forward(self, x):
        return self.net(x)


# Quick check: python models/cnn4layer1.py
if __name__ == "__main__":
    dummy_x = torch.randn(4, 1, 256, 256)
    for cls in (CNN4Steps_Model1, CNN4Steps_Model2, CNN4Steps_Model3,
                CNN4Steps_Model4, CNN4Steps_Model5):
        model = cls()
        steps = sum(not isinstance(layer, nn.Flatten) for layer in model.net)
        widest = max(layer.out_channels for layer in model.net if isinstance(layer, nn.Conv2d))
        params = sum(p.numel() for p in model.parameters())
        print(f"{cls.__name__}: output {tuple(model(dummy_x).shape)}, {steps} steps, "
              f"up to {widest} filters, {params} parameters")