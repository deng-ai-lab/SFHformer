import torch
import torch.nn as nn
from einops import rearrange
import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, bias=True)

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=True),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed_for_upsample(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96, out_dim=64, kernel_size=None):
        super().__init__()
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_dim * patch_size ** 2, kernel_size=1, bias=False),
            nn.PixelShuffle(patch_size),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class DownSample(nn.Module):
    """
    DownSample: Conv
    B*H*W*C -> B*(H/2)*(W/2)*(2*C)
    """

    def __init__(self, input_dim, output_dim, kernel_size=4, stride=2):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = output_dim

        self.proj = nn.Sequential(nn.Conv2d(input_dim, input_dim * 2, kernel_size=2, stride=2))

    def forward(self, x):
        x = self.proj(x)
        return x


class OurFFN(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(OurFFN, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2
        # PW first or DW first?
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim*2, 1),
        )

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1,
                      groups=self.dim_sp),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=5, padding=2,
                      groups=self.dim_sp),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=7, padding=3,
                      groups=self.dim_sp),
        )

        self.gelu = nn.GELU()
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
        )


    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim_sp, dim=1))
        x[1] = self.conv1_1(x[1])
        x[2] = self.conv1_2(x[2])
        x[3] = self.conv1_3(x[3])
        x = torch.cat(x, dim=1)
        x = self.gelu(x)
        x = self.conv_fina(x)


        return x




class OurTokenMixer_For_Local(nn.Module):
    def __init__(
            self,
            dim,
            kernel_size=[1,3,5,7],
            se_ratio=4,
            local_size=8,
            scale_ratio=2,
            spilt_num=4
    ):
        super(OurTokenMixer_For_Local, self).__init__()
        self.dim = dim
        self.c_down_ratio = se_ratio
        self.size = local_size
        self.dim_sp = dim//2

        self.CDilated_1 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, dilation=1, groups=self.dim_sp)
        self.CDilated_2 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=2, dilation=2, groups=self.dim_sp)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        cd1 = self.CDilated_1(x1)
        cd2 = self.CDilated_2(x2)
        x = torch.cat([cd1, cd2], dim=1)

        return x


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels * 2)
        self.bn2 = torch.nn.BatchNorm2d(out_channels * 2)
        self.norm1 = LayerNorm(out_channels * 2, 'WithBias')

        self.conv_layer1 = nn.Sequential(torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                                        kernel_size=1, stride=1, padding=0, groups=self.groups,bias=True),
                                        nn.GELU(),
                                        )


        self.conv_layer2 = nn.Sequential(
                                        nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3,
                                        padding=1, stride=1, groups=in_channels * 2,bias=True),
                                        )

    def forward(self, x):
        batch, c, h, w = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = self.bn1(ffted)
        ffted = self.conv_layer2(ffted) + ffted
        ffted = self.conv_layer1(ffted)  # (batch, c*2, h, w/2+1)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.view_as_complex(ffted)

        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output


class OurTokenMixer_For_Gloal(nn.Module):
    def __init__(
            self,
            dim,
            kernel_size=[1,3,5,7],
            se_ratio=4,
            local_size=8,
            scale_ratio=2,
            spilt_num=4
    ):
        super(OurTokenMixer_For_Gloal, self).__init__()
        self.dim = dim
        self.c_down_ratio = se_ratio
        self.size = local_size
        self.dim_sp = dim*scale_ratio//spilt_num
        # PW first or DW first?
        # self.conv_init = nn.Sequential(  # PW->DW->
        #     nn.Conv2d(dim, dim*2, 1),
        #     nn.GELU()
        # )
        # self.conv_fina = nn.Sequential(
        #     nn.Conv2d(dim*2, dim, 1),
        #     nn.GELU()
        # )
        self.FFC = FourierUnit(self.dim, self.dim)
        # self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        # x = self.conv_init(x)
        x0 = x
        x = self.FFC(x)
        # x = self.conv_fina(x+x0)
        x = x + x0

        return x


class OurMixer(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer_for_local=OurTokenMixer_For_Local,
            token_mixer_for_gloal=OurTokenMixer_For_Gloal,
            mixer_kernel_size=[1,3,5,7],
            local_size=8,
            flag=None
    ):
        super(OurMixer, self).__init__()
        self.dim = dim
        self.flag = flag
        self.mixer_local = token_mixer_for_local(dim=self.dim,)
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim, kernel_size=mixer_kernel_size,
                                 se_ratio=8, local_size=local_size)

        self.ca_conv = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2*dim, 2*dim//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*dim//2, 2*dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.gelu = nn.GELU()
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, 2*dim, 1),
        )



    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local = self.mixer_local(x[0])
        x_gloal = self.mixer_gloal(x[1])
        x = torch.cat([x_local, x_gloal], dim=1)
        x = self.gelu(x)
        x = self.ca(x) * x
        x = self.ca_conv(x)



        return x


class OurBlock(nn.Module):
    def __init__(
            self,
            dim,
            norm_layer=nn.BatchNorm2d,
            token_mixer=OurMixer,
            kernel_size=[1,3,5,7],
            local_size=8
    ):
        super(OurBlock, self).__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mixer = token_mixer(dim=self.dim, mixer_kernel_size=kernel_size, local_size=local_size, flag=flag)
        self.ffn = OurFFN(dim=self.dim)

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        copy = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x * self.beta + copy

        copy = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x * self.gamma + copy

        return x



# need drop_path?
class OurStage(nn.Module):
    def __init__(
            self,
            depth=int,
            in_channels=int,
            mixer_kernel_size=[1,3,5,7],
            local_size=8
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(OurStage, self).__init__()
        # Init blocks
        self.blocks = nn.Sequential(*[
                OurBlock(
                    dim=in_channels,
                    norm_layer=nn.BatchNorm2d,
                    token_mixer=OurMixer,
                    kernel_size=mixer_kernel_size,
                    local_size=local_size,
                )
            # BaselineBlock(in_channels)
            for index in range(depth)
        ])

    def forward(self, input=torch.Tensor) -> torch.Tensor:
        output = self.blocks(input)
        return output


class Backbone_new(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, patch_size=1,
                 embed_dim=[48, 96, 192, 96, 48], depth=[2, 2, 2, 2, 2],
                 local_size=[4, 4, 4, 4 ,4],
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 norm_layer_transformer=nn.LayerNorm, embed_kernel_size=3,
                 downsample_kernel_size=None, upsample_kernel_size=None):
        super(Backbone_new, self).__init__()

        self.patch_size = patch_size
        if downsample_kernel_size is None:
            downsample_kernel_size = 4
        if upsample_kernel_size is None:
            upsample_kernel_size = 4

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dim[0], kernel_size=embed_kernel_size)
        self.layer1 = OurStage(depth=depth[0], in_channels=embed_dim[0],
                               mixer_kernel_size=[1, 3, 5, 7],  local_size=local_size[0])
        self.skip1 = nn.Conv2d(embed_dim[1], embed_dim[0], 1)
        self.downsample1 = DownSample(input_dim=embed_dim[0], output_dim=embed_dim[1],
                                      kernel_size=downsample_kernel_size, stride=2)
        self.layer2 = OurStage(depth=depth[1], in_channels=embed_dim[1],
                               mixer_kernel_size=[1, 3, 5, 7], local_size=local_size[1])
        self.skip2 = nn.Conv2d(embed_dim[2], embed_dim[1], 1)
        self.downsample2 = DownSample(input_dim=embed_dim[1], output_dim=embed_dim[2],
                                      kernel_size=downsample_kernel_size, stride=2)
        self.layer3 = OurStage(depth=depth[2], in_channels=embed_dim[2],
                               mixer_kernel_size=[1, 3, 5, 7], local_size=local_size[1])
        self.upsample3 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[6],
                                                   out_dim=embed_dim[7])
        self.layer8 = OurStage(depth=depth[7], in_channels=embed_dim[7],
                               mixer_kernel_size=[1, 3, 5, 7], local_size=local_size[3])
        self.upsample4 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[7],
                                                   out_dim=embed_dim[8])
        self.layer9 = OurStage(depth=depth[8], in_channels=embed_dim[8],
                               mixer_kernel_size=[1, 3, 5, 7], local_size=local_size[4])
        self.patch_unembed = PatchUnEmbed(patch_size=patch_size, out_chans=out_chans,
                                          embed_dim=embed_dim[8], kernel_size=3)

    def forward(self, x):
        copy0 = x
        x = self.patch_embed(x)
        x = self.layer1(x)
        copy1 = x

        x = self.downsample1(x)
        x = self.layer2(x)
        copy2 = x

        x = self.downsample2(x)
        x = self.layer3(x)
        x = self.upsample3(x)

        x = self.skip2(torch.cat([x, copy2], dim=1))
        x = self.layer8(x)
        x = self.upsample4(x)

        x = self.skip1(torch.cat([x, copy1], dim=1))
        x = self.layer9(x)
        x = self.patch_unembed(x)

        x = copy0 + x
        return x



def sfhformer_motion_blur():
    return Backbone_new(
        patch_size=1,
        embed_dim=[64, 128, 256, 512, 1024, 512, 256, 128, 64],
        depth=[6, 6, 12, 0, 0, 0, 0, 6, 6],
        local_size=[4, 4, 4, 4, 4],
        embed_kernel_size=3
    )



