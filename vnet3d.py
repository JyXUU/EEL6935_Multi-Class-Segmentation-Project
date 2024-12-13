import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.prelu = nn.PReLU()
        self.norm = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.prelu(x)
        return x

class Conv3DResidual(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super(Conv3DResidual, self).__init__()
        self.convs = nn.ModuleList(
            [Conv3DBlock(in_channels if i == 0 else out_channels, out_channels) for i in range(num_convs)]
        )
        self.skip_connection = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skip = self.skip_connection(x)
        for conv in self.convs:
            x = conv(x)
        return x + skip

class DeConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeConv3D, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = Conv3DBlock(out_channels + out_channels // 2, out_channels)

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        x = torch.cat([lhs, rhs_up], dim=1)
        return self.conv(x)

class VNet3D(nn.Module):
    def __init__(self, n_classes):
        super(VNet3D, self).__init__()
        self.conv_1 = Conv3DResidual(1, 16, 1)
        self.pool_1 = nn.Conv3d(16, 32, kernel_size=2, stride=2)
        self.conv_2 = Conv3DResidual(32, 32, 2)
        self.pool_2 = nn.Conv3d(32, 64, kernel_size=2, stride=2)
        self.conv_3 = Conv3DResidual(64, 64, 3)
        self.pool_3 = nn.Conv3d(64, 128, kernel_size=2, stride=2)
        self.conv_4 = Conv3DResidual(128, 128, 3)
        self.pool_4 = nn.Conv3d(128, 256, kernel_size=2, stride=2)

        self.bottom = Conv3DResidual(256, 256, 3)

        self.deconv_4 = DeConv3D(256, 256)
        self.deconv_3 = DeConv3D(256, 128)
        self.deconv_2 = DeConv3D(128, 64)
        self.deconv_1 = DeConv3D(64, 32)

        self.out = nn.Conv3d(32, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv_1 = self.conv_1(x)
        pool = self.pool_1(conv_1)
        conv_2 = self.conv_2(pool)
        pool = self.pool_2(conv_2)
        conv_3 = self.conv_3(pool)
        pool = self.pool_3(conv_3)
        conv_4 = self.conv_4(pool)
        pool = self.pool_4(conv_4)
        bottom = self.bottom(pool)
        deconv = self.deconv_4(conv_4, bottom)
        deconv = self.deconv_3(conv_3, deconv)
        deconv = self.deconv_2(conv_2, deconv)
        deconv = self.deconv_1(conv_1, deconv)
        return self.sigmoid(self.out(deconv))

if __name__ == "__main__":
    input_shape = (1, 1, 128, 128, 128)  # Batch size, Channels, Depth, Height, Width
    model = VNet3D(n_classes=24)
    x = torch.randn(input_shape)
    output = model(x)
    print(output.shape)  
