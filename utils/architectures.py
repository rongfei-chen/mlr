import torch.nn as nn


class ConvAutoEncoder(nn.Module):
    def __init__(self, height, width, representation_dim=100, first_channels=32,
                 kernel_size=5, stride=1, padding=2):
        super(ConvAutoEncoder, self).__init__()

        self.num_cnn_layers = 4
        self.cnn_channels = 2
        self.height = height
        self.width = width
        self.first_channels = first_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.representation_dim = representation_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, first_channels, kernel_size),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(first_channels, self.cnn_channels * first_channels, kernel_size),
            nn.BatchNorm2d(self.cnn_channels * first_channels),
            nn.ReLU(),
            nn.Conv2d(self.cnn_channels * first_channels, representation_dim, 3)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(representation_dim, self.cnn_channels * first_channels, 3),
            nn.BatchNorm2d(self.cnn_channels * first_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(self.cnn_channels * first_channels, first_channels, kernel_size),
            nn.BatchNorm2d(first_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(first_channels, 1, kernel_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x