"""The same convolutional architecture is used for every latent dimension."""

from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        channels = (1, 8, 16, 32, 32)
        encoder = []
        for cin, cout in zip(channels[:-1], channels[1:]):
            encoder.extend([nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.ReLU()])
        self.convolutions = nn.Sequential(*encoder)
        self.to_code = nn.Linear(8192, dimension)
        self.from_code = nn.Linear(dimension, 8192)
        decoder = []
        channels = (32, 32, 16, 8, 1)
        for index, (cin, cout) in enumerate(zip(channels[:-1], channels[1:])):
            decoder.extend([nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(cin, cout, 3, stride=1, padding=1)])
            if index < 3:
                decoder.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder)

    def encode(self, maps):
        return self.to_code(self.convolutions(maps).flatten(1))

    def decode(self, codes):
        # Dense bottleneck layers and the reconstruction are deliberately linear.
        return self.decoder(self.from_code(codes).reshape(-1, 32, 16, 16))

    def forward(self, maps):
        return self.decode(self.encode(maps))
