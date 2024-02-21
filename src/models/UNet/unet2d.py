from torch import nn
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
from src.models.UNet.building_blocks import *
import torch
import torchvision.transforms.functional as TF


class UNet(nn.Module):

    def __init__(self, in_channels=3, classes=1, layers=None, dropout_p=0):
        super(UNet, self).__init__()
        if layers is None:
            layers = [64, 128, 256]
        layer_layout = [in_channels]
        layer_layout.extend(layers)
        self.layers = layer_layout

        self.double_conv_downs = nn.ModuleList(
            [self.__double_conv(layer, layer_n) for layer, layer_n in zip(self.layers[:-1], self.layers[1:])])

        self.up_trans = nn.ModuleList(
            [nn.ConvTranspose2d(layer, layer_n, kernel_size=2, stride=2)
             for layer, layer_n in zip(self.layers[::-1][:-2], self.layers[::-1][1:-1])])

        self.double_conv_ups = nn.ModuleList(
            [self.__double_conv(layer, layer // 2) for layer in self.layers[::-1][:-2]])

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = dropout_p

        self.final_conv = nn.Conv2d(layers[0], classes, kernel_size=1)


    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout)
        )
        return conv

    def forward(self, x):
        # down layers
        concat_layers = []

        for down in self.double_conv_downs:
            x = down(x)
            if down != self.double_conv_downs[-1]:
                concat_layers.append(x)
                x = self.max_pool_2x2(x)

        concat_layers = concat_layers[::-1]

        # up layers
        for up_trans, double_conv_up, concat_layer in zip(self.up_trans, self.double_conv_ups, concat_layers):
            x = up_trans(x)
            if x.shape != concat_layer.shape:
                x = TF.resize(x, concat_layer.shape[2:])

            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)

        x = self.final_conv(x)

        return x

    # class UNet(Module):
# 
#     def __init__(self,
#                  outSize, encChannels=(3, 64, 128, 256),
#                  decChannels=(256, 128, 64),
#                  nbClasses=4, retainDim=True):
#         super().__init__()
#         # initialize the encoder and decoder
#         self.encoder = Encoder(encChannels)
#         self.decoder = Decoder(decChannels)
#         # initialize the regression head and store the class variables
#         self.head = Conv2d(decChannels[-1], nbClasses, 1)
#         self.retainDim = retainDim
#         self.outSize = outSize
# 
#     def forward(self, x):
#         # grab the features from the encoder
#         encFeatures = self.encoder(x)
#         # pass the encoder features through decoder making sure that
#         # their dimensions are suited for concatenation
#         decFeatures = self.decoder(encFeatures[::-1][0],
#                                        encFeatures[::-1][1:])
#         # pass the decoder features through the regression head to
#         # obtain the segmentation mask
#         map = self.head(decFeatures)
#         # check to see if we are retaining the original output
#         # dimensions and if so, then resize the output to match them
# 
#         if self.retainDim:
#             map = F.interpolate(map, self.outSize)
#         # return the segmentation map, where the returned output is a softmax regression
#         return map