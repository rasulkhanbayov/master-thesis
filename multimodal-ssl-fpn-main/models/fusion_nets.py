import torch
from torch import nn

from config import config
# Lachinot et al.
from models.lachinovnet.model import CustomNet
from models.lachinovnet import blocks
# ReSensNet model
from models.resensnet import ModifiedUnet3D
from factory_utils import get_factory_adder



add_class, model_factory = get_factory_adder()


@add_class
class LachinovNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [1, 1, 2, 4]
        number_of_channels = [int(32 * 1 * 2 ** i) for i in range(0, len(layers))]
        self.model = CustomNet(
            depth=len(layers),
            encoder_layers=layers,
            number_of_channels=number_of_channels,
            number_of_outputs=config.num_outputs,
            block=blocks.Residual_bottleneck_bn
        )

    def forward(self, x):
        return self.model(x)


@add_class
class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.resensnet = ModifiedUnet3D()

    def forward(self, x):
        # Z x W x H
        oct = x['image'].permute(0,1,2,4,3)
        oct_seg = self.resensnet(oct)
        oct_seg = oct_seg.permute(0,1,2,4,3)

        # scale_factor = 5
        # oct_seg = oct_seg * scale_factor

        # seg = oct_seg - oct_seg.min()
        # seg = seg / seg.max()

        seg = torch.sigmoid(oct_seg)
        # seg = torch.softmax(oct_seg, 3)
        return {
            'prediction': seg,
        }
    
    # def forward(self, x):
    #     # Z x W x H
    #     oct = x['image'].permute(0,1,2,4,3)
    #     oct_seg = self.resensnet(oct)
    #     oct_seg = oct_seg.permute(0,1,2,4,3)
        
    #     seg = oct_seg - oct_seg.min()  # Shift min to 0
    #     seg = seg / seg.max()  # Normalize to 0–1
    #     seg = seg * 255  # Scale to 0–255 for visualization
    #     seg = seg.clamp(0, 255)  # Ensure valid pixel range

    #     # print(oct_seg.min(), oct_seg.max())

    #     return {
    #         'prediction': seg,
    #     }


@add_class
class ReSensNet(FPN):
    def __init__(self):
        super().__init__()
        self.resensnet = ModifiedUnet3D(is_original=True)
