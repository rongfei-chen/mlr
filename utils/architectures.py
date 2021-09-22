import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, height, width, representation_dim=100, first_channels=32,
                 kernel_size=5, stride=1, padding=2):
        super(ConvEncoder, self).__init__()

        self.num_cnn_layers = 4
        self.cnn_channels = 2
        self.height = height
        self.width = width
        self.first_channels = first_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.representation_dim = representation_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, first_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(first_channels),
            nn.GELU())
        self.pool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.conv2 = nn.Sequential(
            nn.Conv2d(first_channels, self.cnn_channels * first_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(self.cnn_channels * first_channels),
            nn.GELU())
        self.pool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.cnn_channels * first_channels, (self.cnn_channels ** 2) * first_channels, 3, stride=stride, padding=padding),
            nn.BatchNorm2d((self.cnn_channels ** 2) * first_channels),
            nn.GELU())
        self.pool3 = nn.MaxPool2d(kernel_size=3, return_indices=True)

        self.conv4 = nn.Conv2d(
            (self.cnn_channels ** 2) * first_channels, representation_dim, 3, stride=stride, padding=padding)
        self.pool4 = nn.MaxPool2d(kernel_size=4, return_indices=True)

    def forward(self, x):
        x = self.conv1(x)
        size_1 = x.size()
        x, idx_1 = self.pool1(x)
        x = self.conv2(x)
        size_2 = x.size()
        x, idx_2 = self.pool2(x)
        x = self.conv3(x)
        size_3 = x.size()
        x, idx_3 = self.pool3(x)
        x = self.conv4(x)
        size_4 = x.size()
        x, idx_4 = self.pool4(x)

        return x, [idx_4, idx_3, idx_2, idx_1], [size_4, size_3, size_2, size_1]

    def calc_out_size(self):
        height = int(self.height / 16)
        width = int(self.width / 16)
        kernels = (self.cnn_channels ** (self.num_cnn_layers - 1)) * \
                  self.first_channels
        return kernels * height * width


class ConvDecoder(nn.Module):
    def __init__(self, height, width, representation_dim=100, first_channels=32,
                 kernel_size=5, stride=1, padding=2):
        super(ConvDecoder, self).__init__()

        self.num_cnn_layers = 4
        self.cnn_channels = 2
        self.height = height
        self.width = width
        self.first_channels = first_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.representation_dim = representation_dim

        self.unpool0 = nn.MaxUnpool2d(kernel_size=4)

        self.unconv1 = nn.Sequential(
            nn.ConvTranspose2d(representation_dim, (self.cnn_channels ** 2) * first_channels, 3,
                           stride=stride, padding=padding),
            nn.BatchNorm2d((self.cnn_channels ** 2) * first_channels),
            nn.GELU())
        self.unpool1 = nn.MaxUnpool2d(kernel_size=3)

        self.unconv2 = nn.Sequential(
            nn.ConvTranspose2d((self.cnn_channels ** 2) * first_channels, self.cnn_channels * first_channels, 3,
                           stride=stride, padding=padding),
            nn.BatchNorm2d(self.cnn_channels * first_channels),
            nn.GELU())
        self. unpool2 = nn.MaxUnpool2d(kernel_size=2)

        self.unconv3 = nn.Sequential(
            nn.ConvTranspose2d(self.cnn_channels * first_channels, first_channels, kernel_size, stride=stride,
                           padding=padding),
            nn.BatchNorm2d(first_channels),
            nn.GELU())
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2)

        self.unconv4 = nn.Sequential(
            nn.ConvTranspose2d(first_channels, 1, kernel_size, stride=stride, padding=padding),
            nn.Sigmoid())

    def forward(self, x, pool_indices, pool_sizes):
        x = self.unpool0(x, pool_indices[0], output_size=pool_sizes[0])
        x = self.unconv1(x)
        x = self.unpool1(x, pool_indices[1], output_size=pool_sizes[1])
        x = self.unconv2(x)
        x = self.unpool2(x, pool_indices[2], output_size=pool_sizes[2])
        x = self.unconv3(x)
        x = self.unpool3(x, pool_indices[3], output_size=pool_sizes[3])
        x = self.unconv4(x)

        return x

    def calc_out_size(self):
        height = int(self.height / 16)
        width = int(self.width / 16)
        kernels = (self.cnn_channels ** (self.num_cnn_layers - 1)) * \
                  self.first_channels
        return kernels * height * width


class ConvAutoEncoder(nn.Module):
    def __init__(self, height, width, representation_dim=25, first_channels=32,
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

        self.encoder = ConvEncoder(height, width, representation_dim=representation_dim, first_channels=first_channels,
                 kernel_size=kernel_size, stride=stride, padding=padding)
        self.decoder = ConvDecoder(height, width, representation_dim=representation_dim, first_channels=first_channels,
                 kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x, pool_indices, pool_sizes = self.encoder(x)
        representation = x.view(x.size(0), -1)
        x = self.decoder(x, pool_indices, pool_sizes)
        return x, representation

    def calc_out_size(self):
        height = int(self.height / 16)
        width = int(self.width / 16)
        kernels = (self.cnn_channels ** (self.num_cnn_layers - 1)) * \
                  self.first_channels
        return kernels * height * width
