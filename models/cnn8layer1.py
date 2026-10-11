"""
8-step CNNs.

Two architectures:
    A: two conv blocks, BatchNorm in the first
    B: three conv layers with max-pools between them

    Narrow:  Model1 (A, 16 -> 32),   Model2 (B, 16 -> 32 -> 64)
    Wide:    Model3 (A, 32 -> 128),  Model4 (B, 32 -> 64 -> 128), Model5 (B, 64 -> 128 -> 256)

Models 2, 4 and 5 are the same network at 1x, 2x and 4x the width.
"""
import torch
import torch.nn as nn


class CNN8Steps_Model1(nn.Module):
    """Narrow baseline: two 3x3 conv blocks (16 -> 32), BatchNorm in the first."""

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),     # Step 1: Conv (256x256)
            nn.BatchNorm2d(16),                             # Step 2: Normalisation
            nn.LeakyReLU(0.2),                              # Step 3: Activation
            nn.MaxPool2d(4, 4),                             # Step 4: Pool (shrinks to 64x64)

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),    # Step 5: Conv (64x64)
            nn.LeakyReLU(0.2),                              # Step 6: Activation
            nn.AdaptiveAvgPool2d((2, 2)),                   # Step 7: Pool (shrinks to 2x2)

            # Decision
            nn.Flatten(),                                   # 32 * 2 * 2 = 128 values
            nn.Linear(32 * 2 * 2, 2),                       # Step 8: Output (Omega_m, sigma_8)
        )

    def forward(self, x):
        return self.net(x)


class CNN8Steps_Model2(nn.Module):
    """Narrow: three 3x3 conv layers (16 -> 32 -> 64) with max-pools between them."""

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # Feature extraction
            nn.Conv2d(1, 16, kernel_size=3, padding=1),     # Step 1: Conv (256x256)
            nn.MaxPool2d(4, 4),                             # Step 2: Pool (shrinks to 64x64)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),    # Step 3: Conv (64x64)
            nn.MaxPool2d(4, 4),                             # Step 4: Pool (shrinks to 16x16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # Step 5: Conv (16x16)
            nn.LeakyReLU(0.2),                              # Step 6: Activation
            nn.AdaptiveAvgPool2d((2, 2)),                   # Step 7: Pool (shrinks to 2x2)

            # Decision
            nn.Flatten(),                                   # 64 * 2 * 2 = 256 values
            nn.Linear(64 * 2 * 2, 2),                       # Step 8: Output
        )

    def forward(self, x):
        return self.net(x)


class CNN8Steps_Model3(nn.Module):
    """Wide: Model1 (two conv blocks) with 32 -> 128 filters."""

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),     # Step 1: Conv (256x256)
            nn.BatchNorm2d(32),                             # Step 2: Normalisation
            nn.LeakyReLU(0.2),                              # Step 3: Activation
            nn.MaxPool2d(4, 4),                             # Step 4: Pool (shrinks to 64x64)

            # Block 2
            nn.Conv2d(32, 128, kernel_size=3, padding=1),   # Step 5: Conv (64x64)
            nn.LeakyReLU(0.2),                              # Step 6: Activation
            nn.AdaptiveAvgPool2d((2, 2)),                   # Step 7: Pool (shrinks to 2x2)

            # Decision
            nn.Flatten(),                                   # 128 * 2 * 2 = 512 values
            nn.Linear(128 * 2 * 2, 2),                      # Step 8: Output
        )

    def forward(self, x):
        return self.net(x)


class CNN8Steps_Model4(nn.Module):
    """Wide: Model2 (three conv layers) with 32 -> 64 -> 128 filters."""

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # Feature extraction
            nn.Conv2d(1, 32, kernel_size=3, padding=1),     # Step 1: Conv (256x256)
            nn.MaxPool2d(4, 4),                             # Step 2: Pool (shrinks to 64x64)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # Step 3: Conv (64x64)
            nn.MaxPool2d(4, 4),                             # Step 4: Pool (shrinks to 16x16)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # Step 5: Conv (16x16)
            nn.LeakyReLU(0.2),                              # Step 6: Activation
            nn.AdaptiveAvgPool2d((2, 2)),                   # Step 7: Pool (shrinks to 2x2)

            # Decision
            nn.Flatten(),                                   # 128 * 2 * 2 = 512 values
            nn.Linear(128 * 2 * 2, 2),                      # Step 8: Output
        )

    def forward(self, x):
        return self.net(x)


class CNN8Steps_Model5(nn.Module):
    """Widest: Model2 (three conv layers) with 64 -> 128 -> 256 filters."""

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # Feature extraction
            nn.Conv2d(1, 64, kernel_size=3, padding=1),     # Step 1: Conv (256x256)
            nn.MaxPool2d(4, 4),                             # Step 2: Pool (shrinks to 64x64)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # Step 3: Conv (64x64)
            nn.MaxPool2d(4, 4),                             # Step 4: Pool (shrinks to 16x16)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Step 5: Conv (16x16)
            nn.LeakyReLU(0.2),                              # Step 6: Activation
            nn.AdaptiveAvgPool2d((2, 2)),                   # Step 7: Pool (shrinks to 2x2)

            # Decision
            nn.Flatten(),                                   # 256 * 2 * 2 = 1024 values
            nn.Linear(256 * 2 * 2, 2),                      # Step 8: Output
        )

    def forward(self, x):
        return self.net(x)


# Quick check: python models/cnn8layer1.py
if __name__ == "__main__":
    dummy_x = torch.randn(4, 1, 256, 256)
    for cls in (CNN8Steps_Model1, CNN8Steps_Model2, CNN8Steps_Model3,
                CNN8Steps_Model4, CNN8Steps_Model5):
        model = cls()
        steps = sum(not isinstance(layer, nn.Flatten) for layer in model.net)
        widest = max(layer.out_channels for layer in model.net if isinstance(layer, nn.Conv2d))
        params = sum(p.numel() for p in model.parameters())
        print(f"{cls.__name__}: output {tuple(model(dummy_x).shape)}, {steps} steps, "
              f"up to {widest} filters, {params} parameters")