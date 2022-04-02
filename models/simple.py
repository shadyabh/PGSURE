import torch
import torch.nn as nn


# class UNet(nn.Module):
#     def __init__(self, in_channels, out_channels=3, base_channels=16):
#         super(UNet, self).__init__()

#         self.d1 = nn.Sequential(
#             nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
#             nn.LeakyReLU(),
#             nn.AvgPool2d(2, 2)
#         )
#         self.d2 = nn.Sequential(
#             nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
#             nn.LeakyReLU(),
#             nn.AvgPool2d(2, 2)
#         )
#         self.d3 = nn.Sequential(
#             nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
#             nn.LeakyReLU(),
#             nn.AvgPool2d(2, 2)
#         )

#         self.u3 = nn.Sequential(
#             nn.ConvTranspose2d(base_channels, base_channels, 4),
#             nn.LeakyReLU()
#         )
#         self.u2 = nn.Sequential(
#             nn.ConvTranspose2d(2*base_channels, base_channels, 4),
#             nn.LeakyReLU()
#         )
#         self.u1 = nn.Sequential(
#             nn.ConvTranspose2d(2*base_channels, out_channels, 4),
#             nn.LeakyReLU()
#         )

#     def forward(self, x):


class simple(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=16):
        super(simple, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=5, padding=2, padding_mode='circular'),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Conv2d(base_channels, base_channels, kernel_size=5, padding=2, padding_mode='circular'),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Conv2d(base_channels, base_channels, kernel_size=5, padding=2, padding_mode='circular'),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Conv2d(base_channels, out_channels, kernel_size=5, padding=2, padding_mode='circular')
        )

    def forward(self, x):
        return self.layers(x)