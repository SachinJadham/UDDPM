import torch
from torch import nn
import math


# ============================================ TimeEmbedding ==================================== #
class TimeEmbedding(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        assert out_channels % 2 == 0  # Embedding dimensions must be even
        self.out_dims = out_channels  # Input: (B); Output: (B, out_dims)

    def forward(self, t: torch.Tensor):
        # calculate position embedding;
        # PE_1(t, i) = sin( t / (10000 ^ (i/dim))
        # PE_2(t, i) = cos( t / (10000 ^ (i/dim))
        half_dim = self.out_dims // 2
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        return emb


# ============================================= Down Block ===================================== #
class DownBlock(nn.Module):
    r"""
        Down conv block with attention.
        Sequence of following block
        1. Resnet block with time embedding
            resnet1 -> time emb -> resnet2
        2. Attention block
        3. Downsample using 2x2 average pooling

        https://www.youtube.com/watch?v=vu6eKteJWew
    """

    def __init__(self, in_channels, out_channels, t_emb_dim, num_attentionHeads, down_sample_flag):
        super().__init__()

        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )

        # reshape/project time embedding dims to number of channels of this Resnet block,
        # for the purpose of concatenation in the forward process.
        self.time_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=t_emb_dim, out_features=out_channels)
        )

        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )

        # residual connection (input to Resnet output); 1x1 Conv layer -> just makes sure that dims are correct.
        self.residual_input_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        # Attention
        self.attention_norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_attentionHeads, batch_first=True)

        # Downsampling
        self.down_sample_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                          kernel_size=4, stride=2, padding=1) if down_sample_flag else nn.Identity()

    def forward(self, x, t_emb):
        resnet_input = x  # save input for residual connection below.

        # Resnet Block
        out = self.resnet_conv_first(x)
        out = out + self.time_emb_layers(t_emb)[:, :, None, None]
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)  # residual connection concatenation.

        # Attention Block
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h * w)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)  # transpose ensures that 'channels' are now at last dims. [..., C]
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)  # convert back.
        out = out + out_attn  # residual connection concatenation.

        # Down Sample
        out = self.down_sample_conv(out)

        return out


# ============================================= Mid Block ===================================== #
class MidBlock(nn.Module):
    r"""
        Mid-conv block with attention.
        Sequence of following blocks
        1. Resnet block with time embedding
        2. Attention block
        3. Resnet block with time embedding
    """

    def __init__(self, in_channels, out_channels, t_emb_dim, num_attentionHeads):
        super().__init__()

        # ### Resnet Blocks with time embedding ###
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            )
        ])

        self.time_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            ),
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
        ])

        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        ])

        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        ])

        # ### Attention Block ###
        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads=num_attentionHeads, batch_first=True)

    def forward(self, x, t_emb):
        resnet_input = x  # save input for residual connection below.

        # First Resnet Block
        out = self.resnet_conv_first[0](x)
        out = out + self.time_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        # Attention Block
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h * w)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)  # transpose ensures that 'channels' are now at last dims. [..., C]
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)  # convert back.
        out = out + out_attn  # residual connection concatenation.

        # Second Resnet Block
        resnet_input = out  # save input for residual connection below.
        out = self.resnet_conv_first[1](out)
        out = out + self.time_emb_layers[1](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[1](out)
        out = out + self.residual_input_conv[1](resnet_input)

        return out


# ============================================= Up Block ====================================== #
class UpBlock(nn.Module):
    r"""
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """

    def __init__(self, in_channels, out_channels, t_emb_dim, num_attentionHeads, up_sample_flag):
        super().__init__()

        # Up Sampling
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                                 kernel_size=4, stride=2, padding=1) \
            if up_sample_flag else nn.Identity()

        # residual connection (input to Resnet output); 1x1 Conv layer -> just makes sure that dims are correct.
        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        # reshape/project time embedding dims to number of channels of this Resnet block,
        # for the purpose of concatenation in the forward process.
        self.time_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=t_emb_dim, out_features=out_channels)
        )

        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        # Attention
        self.attention_norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_attentionHeads, batch_first=True)

    def forward(self, x, t_emb, out_down):
        # Up Sample ; you can also (concat -> resnet -> attention -> then up sample) too
        x = self.up_sample_conv(x)  # up sample the input
        x = torch.cat([x, out_down], dim=1)  # concat the corresponding down-block output; skip connections

        # Resnet Block
        resnet_input = x  # save input for residual connection below.
        out = self.resnet_conv_first(x)
        out = out + self.time_emb_layers(t_emb)[:, :, None, None]
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)  # residual connection concatenation.

        # Attention Block
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h * w)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)  # transpose ensures that 'channels' are now at last dims. [..., C]
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)  # convert back.
        out = out + out_attn  # residual connection concatenation.

        return out


# =============================================== MODEL ======================================== #
class Unet(nn.Module):
    def __init__(self, imagechannels: int = 3):
        super().__init__()

        # dimensions
        self.imchannels = imagechannels  # Image channels; 3
        self.down_channels = [32, 64, 128, 256]  # shall be converted to 3 down blocks; ith dim -> (i+1)th dim
        self.mid_channels = [256, 256, 128]  # 2 mid blocks
        self.t_emb_dim = 128  # o/p dim = first down channel dim * 4; 32*4
        self.down_sample = [True, True, False]  # don't downsample the last down block; goes in as down_sample_flag

        # time-embeddings; get initial timestep representations.
        # these layers are only called once in whole forward pass.
        self.time_mlp = nn.Sequential(
            TimeEmbedding(out_channels=self.t_emb_dim),
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.GELU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        # first transformation of image; output dim == input dim to first DownBlock/Encoder.
        # Project image into feature map
        self.image_proj = nn.Conv2d(in_channels=self.imchannels, out_channels=self.down_channels[0],
                                    kernel_size=3, stride=1, padding='same')

        # create down-blocks; based on down_channels
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i + 1],
                                        self.t_emb_dim, num_attentionHeads=4,
                                        down_sample_flag=self.down_sample[i]))

        # create mid-blocks
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1],
                                      self.t_emb_dim, num_attentionHeads=4))

        # create up-blocks
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels)-1)):
            self.ups.append(UpBlock(self.down_channels[i] * 2, self.down_channels[i-1] if i != 0 else 16,
                                    self.t_emb_dim, num_attentionHeads=4, up_sample_flag=self.down_sample[i]))

        # output of last up block; undergoes Normalization, activation(in forward fn) and convolution
        # to get back to same channels as Input.
        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, imagechannels, kernel_size=3, padding=1)

    def forward(self, x, timesteps):

        # x: [Batch, channels, Height, Width]
        # timesteps: [Batch, ]
        t_emb = self.time_mlp(timesteps)  # [B, 128]

        r""" Shapes assuming downblocks are [C1, C2, C3, C4]; [32, 64, 128, 256]
             Shapes assuming downsamples are [True, True, False]
             Shapes assuming midblocks are [C4, C4, C3]; [256, 256, 128]
             Shapes assuming upblocks are [C3, C2, C1, 16 ]; [128, 64, 32, 16]
             Shapes assuming upsamples are [True, True, False]
              """

        # FIRST IMAGE TRANSFORM
        # [B, image_channels, H, W]
        out = self.image_proj(x)
        # [B, C1, H, W]

        # CALL THE DOWN BLOCKS
        down_outs = []  # save the output of down blocks; to be used in up blocks later.
        for down in self.downs:
            # print(out.shape)      # down_outs  [ [B, C1, H, W], [B, C2, H/2, W/2], [B, C3, H/4, W/4] ]
            down_outs.append(out)
            out = down(out, t_emb)
        # out [B, C4, H/4, W/4]  # last index -> no down sample.
        # print(out.shape)

        # CALL THE MID BLOCKS
        for mid in self.mids:
            # print(out.shape)
            out = mid(out, t_emb)
        # out [B, C3, H/4, W/4]
        # print(out.shape)            # torch.Size([B, 128, H/4, W/4])

        # CALL THE UP BLOCKS
        for up in self.ups:
            down_out = down_outs.pop()
            # print(out.shape, down_out.shape)
            out = up(out, t_emb, down_out)
        # out [ [B, C2, H/4, W/4], [B, C1, H/2, W/2], [B, 16, H, W] ]
        # print(out.shape)            # torch.Size([B, 16, H, W])

        # FINAL Norm and conv
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out [B, C, H, W]

        return out
