# Sub Components (encoder, decoder, bottleneck)
import torch
from torch.nn import ModuleList, Sequential
from torchvision.transforms.v2 import CenterCrop

from src.models.Swin_Unet.building_blocks import *


class DownSwin(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, is_intro,
                 heads, head_dim, window_size, relative_position_embedding, drop, mlp_ratio,
                 patch_size=(4, 4), downscale_factor=None):
        super().__init__()

        # Core fields for all components
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.is_intro = is_intro

        # Transformer definitions
        self.heads = heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.relative_position_embedding = relative_position_embedding
        self.drop = drop
        self.mlp_dim = int(hidden_dim * mlp_ratio)

        # Components for just patch embedding or merging
        self.patch_size = patch_size
        self.downscale_factor = downscale_factor

    def forward(self, x):
        if self.is_intro:
            layers = Sequential(
                PatchEmbed(input_dim=self.input_dim, linear_embedding_out=self.hidden_dim, patch_size=self.patch_size),
                SwinTransformerBlock(dim=self.hidden_dim, heads=self.heads, head_dim=self.hidden_dim,
                                     is_shifted=False, window_size=self.window_size,
                                     relative_position_embedding=self.relative_position_embedding,
                                     drop=self.drop, mlp_dim=self.mlp_dim),
                SwinTransformerBlock(dim=self.hidden_dim, heads=self.heads, head_dim=self.hidden_dim,
                                     is_shifted=True, window_size=self.window_size,
                                     relative_position_embedding=self.relative_position_embedding,
                                     drop=self.drop, mlp_dim=self.mlp_dim)
            )
        else:
            layers = Sequential(
                PatchMerging(input_dim=self.input_dim, out_dim=self.hidden_dim, downscale_factor=self.downscale_factor),
                SwinTransformerBlock(dim=self.hidden_dim, heads=self.heads, head_dim=self.hidden_dim,
                                     is_shifted=False, window_size=self.window_size,
                                     relative_position_embedding=self.relative_position_embedding,
                                     drop=self.drop, mlp_dim=self.mlp_dim),
                SwinTransformerBlock(dim=self.hidden_dim, heads=self.heads, head_dim=self.hidden_dim,
                                     is_shifted=True, window_size=self.window_size,
                                     relative_position_embedding=self.relative_position_embedding,
                                     drop=self.drop, mlp_dim=self.mlp_dim)
            )
        return layers(x)


class UpSwin(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 heads, head_dim, window_size, relative_position_embedding, drop, mlp_ratio,
                 downscale_factor):
        super().__init__()

        # Core fields for all components
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Transformer definitions
        self.heads = heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.relative_position_embedding = relative_position_embedding
        self.drop = drop
        self.mlp_dim = int(hidden_dim * mlp_ratio)

        # Components for just patch embedding or merging
        self.downscale_factor = downscale_factor

    def forward(self, x):
        layers = Sequential(
            PatchExpanding(input_dim=self.input_dim, dim=self.hidden_dim, dim_scale=self.downscale_factor),
            SwinTransformerBlock(dim=self.hidden_dim, heads=self.heads, head_dim=self.hidden_dim,
                                 is_shifted=False, window_size=self.window_size,
                                 relative_position_embedding=self.relative_position_embedding,
                                 drop=self.drop, mlp_dim=self.mlp_dim),
            SwinTransformerBlock(dim=self.hidden_dim, heads=self.heads, head_dim=self.hidden_dim,
                                 is_shifted=True, window_size=self.window_size,
                                 relative_position_embedding=self.relative_position_embedding,
                                 drop=self.drop, mlp_dim=self.mlp_dim))
        return layers(x)


class Encoder(torch.nn.Module):
    def __init__(self, channels, heads, head_dim, window_size, relative_position_embedding, drop, mlp_ratio,
                 patch_size=(4, 4), downscale_factor=None):
        super().__init__()
        self.enc_sections = ModuleList(
            [DownSwin(channels[i], channels[i + 1], i == 0, heads=heads, head_dim=head_dim, window_size=window_size,
                      relative_position_embedding=relative_position_embedding, drop=drop,
                      mlp_ratio=mlp_ratio, patch_size=patch_size, downscale_factor=downscale_factor)
             for i in range(len(channels) - 1)])

    def forward(self, x):
        sections = []
        for sec in self.enc_sections:
            x = sec(x)
            sections.append(x)
        return sections


class Decoder(torch.nn.Module):
    def __init__(self, decoder_channels,  heads, head_dim, window_size, relative_position_embedding, drop, mlp_ratio, downscale_factor):
        super().__init__()
        self.decoder_channels = decoder_channels
        self.patch_expanding = ModuleList(
            [PatchExpanding(input_dim=decoder_channels[i], dim=decoder_channels[i + 1], dim_scale=downscale_factor )
             for i in range(len(decoder_channels) - 1)])
        self.swin_blocks = ModuleList(
            [SwinTransformerBlock(dim=decoder_channels[i], heads=heads, head_dim=head_dim,
                                     is_shifted=True, window_size=window_size,
                                     relative_position_embedding=relative_position_embedding,
                                     drop=drop, mlp_dim=mlp_ratio) for i in range(1, len(decoder_channels) - 1)])
        self.s_swin_blocks = ModuleList(
            [SwinTransformerBlock(dim=decoder_channels[i], heads=heads, head_dim=head_dim,
                                  is_shifted=False, window_size=window_size,
                                  relative_position_embedding=relative_position_embedding,
                                  drop=drop, mlp_dim=mlp_ratio) for i in range(1, len(decoder_channels) - 1)])

    def forward(self, x, enc_layers):
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.patch_expanding[i](x)
            # combine sematnic info form encoder with current decoder block
            encFeat = self.crop(enc_layers[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.swin_blocks[i](x)
            x = self.s_swin_blocks[i](x)

        return x

    def crop(self, enc_features, x):
        (_, _, H, W) = x.shape
        enc_features = CenterCrop([H, W])(enc_features)
        return enc_features


# Full code implementation

class SwinUnet(torch.nn.Module):
    def __init__(self, in_channels, classes, depth_layers, heads, head_dim, window_size, embed_dim,
                 relative_position_embedding, drop, mlp_ratio, downscale_factor):
        super(SwinUnet, self).__init__()

        # set up depth layers
        layers = [in_channels]
        layers.extend(depth_layers)
        self.depth_layers = layers

        self.encoder = Encoder(depth_layers, heads, head_dim, window_size, relative_position_embedding, drop, mlp_ratio,
                               downscale_factor)
        self.decoder = Decoder(depth_layers[::-1], heads, head_dim, window_size, relative_position_embedding, drop,
                               mlp_ratio, downscale_factor)
        self.final_patch_expand = FinalPatchExpanding(in_dim=(in_channels[0]//window_size, in_channels[1]//window_size),
                                                      dim=embed_dim, dim_scale=4)
        self.final = Conv2d(in_channels=embed_dim, out_channels=classes, kernel_size=1, bias=False)

    def forward(self, x):
        encoder_features = self.encoder(x)
        descend_features = self.decoder(encoder_features[::-1][0], encoder_features[::-1][1:])
        final = self.final(self.final_patch_expand(descend_features))
        return final
