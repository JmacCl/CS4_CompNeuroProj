# Swin-Unet non-transformer major components
import numpy as np
import torch
# Imports
from torch.nn import Linear, GELU, Conv2d, LayerNorm, Parameter, Unfold, Dropout, Identity, ModuleList

from einops import rearrange


# Swin Transfomer Block Implementations

class ResidualBlock(torch.nn.Module):

  def __init__(self, fn):
    super().__init__()
    self.fn = fn

  def forward(self, x, **kwargs):
    return self.fn(x, **kwargs) + x


class Normalization(torch.nn.Module):

  def __init__(self, dim, fn):
    super().__init__()
    self.norm = LayerNorm(dim)
    self.fn = fn

  def forward(self, x):
    return self.fn(self.norm(x))


class MLP(torch.nn.Module):
  def __init__(self, input, hidden_size, output_size, drop_p):
    super().__init__()
    self.linear_one = Linear(input, hidden_size)
    self.gelu = GELU()
    self.linear_two = Linear(hidden_size, output_size)
    self.drop(drop_p)

  def __forward__(self, x):
    x = self.linear_one(x)
    x = self.gelu(x)
    x = self.linear_two(x)
    x = self.drop(x)
    return x

class CyclicShift(torch.nn.Module):

  def __init__(self, displacement):
    super().__init__()
    self.displacement = displacement

  def forward(self, x):

    return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))

def create_mask(window_size, displacement, upper_lower, left_right):
  """
  Create matrix mask mappings inorder to cover certain areas of the shifted windows
  """

  window_mask = torch.zeros(window_size**2, window_size**2)

  if upper_lower:
    window_mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
    window_mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

  if left_right:
    window_mask = rearrange(window_mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
    window_mask[:, -displacement:, :, :-displacement] = float('-inf')
    window_mask[:, :-displacement, :, -displacement:] = float('-inf')
    window_mask = rearrange(window_mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

  return window_mask

def get_relative_distances(window_size):
  indicies = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
  distances = indicies[None, :, :] - indicies[:, None, :]
  return distances


class WindowAttention(torch.nn.Module):

  def __init__(self, input_dim, heads, head_dim, is_shifted, window_size, relative_position_embedding, drop):
    """
    Implementation for the window multihead self attention block. Is either shifted of full.
    :param input_dim: number of diemneions ofr ht einput channel dimensions for the shifted window multi-headed attention
    :param heads:
    :param head_dim:
    :param is_shifted:
    :param window_size:
    :param pos_embedding:
    """
    super().__init__()

    self.input_dim = input_dim
    self.heads = heads
    self.attn_scale = head_dim ** -0.5
    self.window_size = window_size
    self.is_shifted = is_shifted
    self.relative_position_embedding = relative_position_embedding


    if self.is_shifted:
      # shift pixels by half the size of the window
      self.displacement = window_size // 2
      self_cyclic_shift = CyclicShift(-self.displacement)
      self.reverse_cyclic_shift = CyclicShift(self.displacement)

      # Create masks to ignore certain sections, should be set to zero grad to ignore training

      self.new_low_mask = Parameter(create_mask(window_size=window_size, displacement=self.displacement,
                                                upper_lower=True, left_right=False), requires_grad=False)

      self.new_right_mask = Parameter(create_mask(window_size=window_size, displacement=self.displacement,
                                                  upper_lower=False, left_right=True), requires_grad=False)

    # Set up QKV encodings
    output_channels = head_dim * input_dim
    self.qkv = Linear(input_dim, output_channels, bias=False)

    if self.relative_position_embedding:
      self.relative_indicies = get_relative_distances(window_size) + window_size + 1
      self.positional_embedding = Parameter(torch.randn(2*window_size - 1, 2*window_size - 1))
    else:
      self.positional_embedding = Parameter(torch.randn(window_size ** 2, window_size ** 2))

    self.output_drop = Dropout(self.output_drop)
    self.attention_output = Linear(output_channels, input_dim)

  def forward(self, x):

    if self.is_shifted:
      # Shift the input if specified
      x = self.cyclic_shift(x)

    # Derive the necessaary dimensions from the input and the heads
    B, H, W, _, heads = *x.shape, self.heads

    qkv = self.qkv(x).chunk(3, dim=-1)

    window_H_count = H//self.window_size
    window_W_count = W//self.window_size

    Q, K, V = map(lambda t: rearrange(t, 'b (whc wh) (wwc ww) (h d) -> b h (whc wwc) (wh ww) d', h=heads, wh=self.window_size, ww=self.window_size ), qkv)

    # Caclulate prdouct with Q and K
    QdotK = torch.einsum('b h w i d, b h w j d -> b h w i j', Q, K) * self.scale

    # Add positional embeddings
    if self.relative_position_embedding:
      ri_row = self.relative_indicies[:, :, 0]
      ri_col = self.relative_indicies[:, :, 1]
      QdotK += self.positional_embedding[ri_row, ri_col]
    else:
      QdotK += self.positional_embedding

    if self.is_shifted:
      # For the last row and last column, mask them
      QdotK[:, :, -window_W_count:] += self.new_low_mask
      QdotK[:, :, window_W_count - 1::window_W_count] += self.new_right_mask

    # Calculate final section of Attention calculation
    soft_comp = QdotK.softmax(dim=-1)
    attention = torch.einsum('b h w i j, b h w j d -> b h w i d', V)

    # Rearange output
    attention = rearrange(attention, ' b h (whc wwc) (wh ww) d -> b (whc wh) (wwc ww) (h d)', h=heads, wh=self.window_size, ww=self.window_size, whc=window_H_count, wwc=window_W_count)
    final = self.attention_output(attention)

    if self.is_shifted:
      final=self.reverse_cyclic_shift(final)

    final = self.output_drop(final)

    return final


class SwinTransformerBlock(torch.nn.Module):
    def __init__(self, dim, heads, head_dim, is_shifted, window_size, relative_position_embedding, drop, mlp_dim):
        super().__init__()
        self.attention_block = ResidualBlock(Normalization(dim, WindowAttention(dim, heads=heads,
                                                                                head_dim=head_dim, is_shifted=is_shifted,
                                                                                window_size=window_size,
                                                                                relative_position_embedding=relative_position_embedding, drop=drop)))

        self.mlp = ResidualBlock(Normalization(dim, MLP(dim, mlp_dim, dim, drop)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp(x)

        return x
class PatchEmbed(torch.nn.Module):
    def __init__(self, input_dim, linear_embedding_out, patch_size=(4, 4)):
      """
      Defines the compontent the partitions the input image into seperate patches, and then linear embeds the
      output into a linear vector
      :param img_size:(I:int, I:int) should a tuple array of integeres that are of the same size
      :return: processed image info
      """
      super().__init__()
      B, C, H, W = input_dim.shape
      self.__process_input((H, W), patch_size)
      self.img_size = (H, W)
      self.patch_size = patch_size
      self.input_channels = C
      self.linear_embedding_out = linear_embedding_out
      self.patches_res = (H[0]/patch_size[0], W/patch_size[1])
      self.partition = Conv2d(C, linear_embedding_out, kernel_size=patch_size, stride=patch_size)
      self.norm = LayerNorm(linear_embedding_out)

    def forward(self, x):
      """
      Defines the component that partitions the input image into separate patches, and then linearly embeds the
      output into a linear vector.
      :param x: Input image tensor
      :return: Processed image information
      """
      B, C, H, W = x.shape
      assert self.img_size[0] == H and self.img_size[1] == W, "Error with input array"
      x = self.partition(x).flatten(2).transpose(1, 2)
      x = self.norm(x)

      return x



    def __process_input(img_size, patch_size):
        assert img_size[0] == img_size[1],  f"Error with img inputs to model, must be same size"
        assert  patch_size[0] ==  patch_size[1],  f"Error with patches inputs to model, must be same size"




class PatchMerging(torch.nn.Module):
  def __init__(self, input_dim, out_dim, downscale_factor):
    super().__init__()
    self.downscale_factor = downscale_factor
    self.patch_merge = Unfold(kernel_size=self.downscale_factor, stride=self.downscale_factor, padding=0)
    self.linear = Linear(input_dim * downscale_factor ** 2, out_dim)

  def forward(self, x):
    B, C, H, W = x.shape
    window_H, window_W = H // self.downscale_factor, W // self.downscale_factor
    x = self.patch_merge(x).view(B, -1, window_H, window_W).permute(0, 2, 3, 1)
    x = self.linear(x)
    return x

class PatchExpanding(torch.nn.Module):
  def __init__(self, input_dim, dim, dim_scale):
    super().init()
    self.input = input_dim
    self.dim = dim
    self.expand = Linear(dim, 2*dim, bias=False) if dim_scale == 2 else Identity()
    self.norm = LayerNorm(dim // dim_scale)

  def forward(self, x):
    H, W = self.input_dim
    x = self.expand(x)
    B, M, C = x.shape
    assert M == H * W, "Error: input features are the wrong size for patch expansion"

    x = x.view(B, H, W, C)
    x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
    x = x.view(B, -1, C//2)
    x= self.norm(x)

    return x

class FinalPatchExpanding(torch.nn.Module):
  def __init__(self, in_dim, dim, dim_scale):
    super().__init__()
    self.input = in_dim
    self.dim = dim
    self.expand = Linear(dim, 2*dim, bias=False) if dim_scale == 2 else Identity()
    self.norm = LayerNorm(dim // dim_scale)

  def forward(self, x):
    H, W = self.in_dim
    x = self.expand(x)
    B, M, C = x.shape
    assert M == H * W, "Error: input features are the wrong size for patch expansion"

    x = x.view(B, H, W, C)
    x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
    x = x.view(B, -1, C//4)
    x = self.norm(x)

    return x


