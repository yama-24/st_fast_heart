import torch
import torch.nn as nn
from torchvision import transforms
from monai.networks.nets import UNet

# 画像の前処理用のトランスフォームを定義
transform = transforms.Compose([
    transforms.ToTensor(),
])

class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.unet = UNet(
                  spatial_dims=2, in_channels=1, out_channels=1,
                  channels=(64, 128, 256, 512, 1024),
                  strides=(2, 2, 2, 2, 2)
        )


    def forward(self, x):
        h = self.unet(x)
        h = h.view(-1, 256, 256)
        return h