from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch

class ConvolutionBlock(Module):
    """
    Defines the class make-up for a given convolution block
    """
    def __init__(self, inChannels, outChannels, kernel_size=3, stride=1):
        """
        Define a convolution block
        :param inChannels:
        :param outChannels:
        """
        super().__init__()
        self.conv1 = Conv2d(inChannels, outChannels, kernel_size=kernel_size, stride=stride)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        """
        Define the forward flow of image information for the block
        :param self:
        :param x: image info
        :return: processed image info
        """
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(Module):
    """
    Define the encoder section of the algorithm
    """

    def __init__(self, channels=(3, 16, 32, 64), max_pool=2):
        """
        Define the system of flow for the encoder
        :param channels: pre-defined flow of information
        """
        super().__init__()
        # Connect the channel mappings of each block to the subsequent block
        self.encBlocks = ModuleList(
            [ConvolutionBlock(channels[i], channels[i + 1])
             for i in range(len(channels) - 1)])
        # add a max pool layer
        self.pool = MaxPool2d(max_pool)

    def forward(self, x):
        """
        Define the forward flow of information
        :param self:
        :param x:
        :return:
        """
        blockOutputs = []
        for block in self.encBlocks:
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        return blockOutputs

class Decoder(Module):

    def __init__(self, channels=(64, 32, 16), ker=2, stride=2):
        super().__init__()
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], kernel_size=ker, stride=stride)
             for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList(
            [ConvolutionBlock(channels[i], channels[i + 1])
             for i in range(len(channels) - 1)])

    def forward(self, x, encFeatures):

        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)
            # combine sematnic info form encoder with current decoder block
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)

        return x

    def crop(self, encFeatures, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        # return the cropped features
        return encFeatures