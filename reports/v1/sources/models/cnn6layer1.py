"""
6-step CNNs.

Narrow (16 / 32 filters):  Model1 (conv block), Model2 (hidden linear layer), Model3 (two convs, 16 -> 32)
Wide (32 -> 128 filters):  Model4 (two convs), Model5 (two convs, 5x5 first)

Model4 is the wide version of Model3. Model5 is Model4 with a 5x5 first conv
"""
import torch
import torch.nn as nn


class CNN6Steps_Model1(nn.Module):
    """Narrow baseline: one full conv block (Conv + BatchNorm + LeakyReLU + MaxPool)."""

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # Feature extraction
            nn.Conv2d(1, 16, kernel_size=3, padding=1),     # Step 1: Conv (256x256)
            nn.BatchNorm2d(16),                             # Step 2: Normalisation
            nn.LeakyReLU(0.2),                              # Step 3: Activation
            nn.MaxPool2d(4, 4),                             # Step 4: Pool (shrinks to 64x64)
            nn.AdaptiveAvgPool2d((2, 2)),                   # Step 5: Pool (shrinks to 2x2)

            # Decision
            nn.Flatten(),                                   # 16 * 2 * 2 = 64 values
            nn.Linear(16 * 2 * 2, 2),                       # Step 6: Output (Omega_m, sigma_8)
        )

    def forward(self, x):
        return self.net(x)


class CNN6Steps_Model2(nn.Module):
    """Narrow: one conv layer, but a hidden linear layer in the decision part."""

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # Feature extraction
            nn.Conv2d(1, 16, kernel_size=3, padding=1),     # Step 1: Conv (256x256)
            nn.LeakyReLU(0.2),                              # Step 2: Activation
            nn.AdaptiveAvgPool2d((2, 2)),                   # Step 3: Pool (shrinks to 2x2)

            # Decision
            nn.Flatten(),                                   # 16 * 2 * 2 = 64 values
            nn.Linear(16 * 2 * 2, 32),                      # Step 4: Hidden linear layer
            nn.LeakyReLU(0.2),                              # Step 5: Activation
            nn.Linear(32, 2),                               # Step 6: Output
        )

    def forward(self, x):
        return self.net(x)


class CNN6Steps_Model3(nn.Module):
    """Narrow: two 3x3 conv layers (16 -> 32) with a max-pool between them."""

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # Feature extraction
            nn.Conv2d(1, 16, kernel_size=3, padding=1),     # Step 1: Conv (256x256)
            nn.MaxPool2d(4, 4),                             # Step 2: Pool (shrinks to 64x64)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),    # Step 3: Conv (64x64)
            nn.LeakyReLU(0.2),                              # Step 4: Activation
            nn.AdaptiveAvgPool2d((2, 2)),                   # Step 5: Pool (shrinks to 2x2)

            # Decision
            nn.Flatten(),                                   # 32 * 2 * 2 = 128 values
            nn.Linear(32 * 2 * 2, 2),                       # Step 6: Output
        )

    def forward(self, x):
        return self.net(x)


class CNN6Steps_Model4(nn.Module):
    """Wide: Model3 (two conv layers) with 32 -> 128 filters."""

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # Feature extraction
            nn.Conv2d(1, 32, kernel_size=3, padding=1),     # Step 1: Conv (256x256)
            nn.MaxPool2d(4, 4),                             # Step 2: Pool (shrinks to 64x64)
            nn.Conv2d(32, 128, kernel_size=3, padding=1),   # Step 3: Conv (64x64)
            nn.LeakyReLU(0.2),                              # Step 4: Activation
            nn.AdaptiveAvgPool2d((2, 2)),                   # Step 5: Pool (shrinks to 2x2)

            # Decision
            nn.Flatten(),                                   # 128 * 2 * 2 = 512 values
            nn.Linear(128 * 2 * 2, 2),                      # Step 6: Output
        )

    def forward(self, x):
        return self.net(x)


class CNN6Steps_Model5(nn.Module):
    """Wide: Model4 with a 5x5 first conv."""

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # Feature extraction
            nn.Conv2d(1, 32, kernel_size=5, padding=2),     # Step 1: Conv (256x256)
            nn.MaxPool2d(4, 4),                             # Step 2: Pool (shrinks to 64x64)
            nn.Conv2d(32, 128, kernel_size=3, padding=1),   # Step 3: Conv (64x64)
            nn.LeakyReLU(0.2),                              # Step 4: Activation
            nn.AdaptiveAvgPool2d((2, 2)),                   # Step 5: Pool (shrinks to 2x2)

            # Decision
            nn.Flatten(),                                   # 128 * 2 * 2 = 512 values
            nn.Linear(128 * 2 * 2, 2),                      # Step 6: Output
        )

    def forward(self, x):
        return self.net(x)


# Quick check: python models/cnn6layer1.py
if __name__ == "__main__":
    dummy_x = torch.randn(4, 1, 256, 256)
    for cls in (CNN6Steps_Model1, CNN6Steps_Model2, CNN6Steps_Model3,
                CNN6Steps_Model4, CNN6Steps_Model5):
        model = cls()
        steps = sum(not isinstance(layer, nn.Flatten) for layer in model.net)
        widest = max(layer.out_channels for layer in model.net if isinstance(layer, nn.Conv2d))
        params = sum(p.numel() for p in model.parameters())
        print(f"{cls.__name__}: output {tuple(model(dummy_x).shape)}, {steps} steps, "
              f"up to {widest} filters, {params} parameters")