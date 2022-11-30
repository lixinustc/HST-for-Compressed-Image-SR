import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from swin_block import RSTB, PatchEmbed, PatchUnEmbed
from timm.models.layers import trunc_normal_


class GRSTB(nn.Module):
    def __init__(self, num_features, img_size, use_embed=True, patch_size=1,
                 depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
                 window_size=7, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, resi_connection='1conv',
                 **kwargs):
        super(GRSTB, self).__init__()
        
        self.num_layers = len(depths)
        patches_resolution = (img_size, img_size)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.mlp_ratio = mlp_ratio
        self.patch_norm = patch_norm
        self.ape = ape
        self.use_embed = use_embed

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=num_features,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection

                         )
            self.layers.append(layer)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=num_features, embed_dim=num_features,
            norm_layer=norm_layer if self.patch_norm else None)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=num_features, embed_dim=num_features,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = norm_layer(num_features)

    def forward(self, x, x_size=None):
        if self.use_embed:
            x_size = (x.shape[2], x.shape[3])
            x = self.patch_embed(x)

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x


class HST(nn.Module):
    def __init__(self, img_size, num_features=[60, 60, 60], scale=4, window_size=8):
        super(HST, self).__init__()
        self.img_size_h = img_size
        self.img_size_m = self.img_size_h // 2
        self.img_size_l = self.img_size_m // 2
        num_fea_h, num_fea_m, num_fea_l = num_features
        self.window_size = window_size
        self.scale = scale

        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        
        self.GRSTB_1 = GRSTB(num_fea_l, self.img_size_l, depths=[6, 6], num_heads=[6, 6], window_size=self.window_size)
        self.conv_after_grstb1 = nn.Conv2d(num_fea_l, num_fea_l, 3, 1, 1)
        self.GRSTB_2 = GRSTB(num_fea_m*2, self.img_size_m, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6], window_size=self.window_size)
        self.conv_after_grstb2 = nn.Conv2d(num_fea_m*2, num_fea_m*2, 3, 1, 1)
        self.GRSTB_3 = GRSTB(num_fea_h*3, self.img_size_h, window_size=self.window_size)
        self.conv_after_grstb3 = nn.Conv2d(num_fea_h*3, num_fea_h*3, 3, 1, 1)

        self.conv_fuse_1 = nn.Conv2d(num_fea_m*2, num_fea_m*2, 1, 1, 0)
        self.conv_fuse_2 = nn.Conv2d(num_fea_h*3, num_fea_h*3, 1, 1, 0)

        self.conv1 = nn.Conv2d(3, num_fea_m, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(num_fea_m, num_fea_l, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(3, num_fea_h, kernel_size=7, stride=1, padding=3)

        self.up = nn.PixelShuffle(2)
        self.upconv1 = nn.Conv2d(num_fea_l, num_fea_m*4, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.Conv2d(num_fea_m*2, num_fea_h*8, kernel_size=3, stride=1, padding=1)
        
        self.conv_before_up = nn.Sequential(nn.Conv2d(num_fea_h*3, 64, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
        self.upsample = nn.Sequential(
                 nn.Conv2d(64, 4 * 64, kernel_size=3, padding=1),
                 nn.PixelShuffle(2),
                 nn.Conv2d(64, 4 * 64, kernel_size=3, padding=1),
                 nn.PixelShuffle(2)
            )

        self.conv_last = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x, t):
        _, _, h, w = x.size()
        # t = self.window_size * self.scale
        mod_pad_h = (t - h % t) % t
        mod_pad_w = (t - w % t) % t
        # mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        # mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'replicate')
        return x

    def forward(self, x):
        _, _, H, W = x.size()
        x = self.check_image_size(x, 8)
        self.mean = self.mean.type_as(x)
        x = x - self.mean
        x_m = self.conv1(x)
        x_l = self.conv2(x_m)
        x_h = self.conv3(x)
        res_l = x_l

        x_l = self.GRSTB_1(x_l)
        x_l = self.conv_after_grstb1(x_l)

        x_l += res_l
        x_l = self.upconv1(x_l)
        x_l = self.up(x_l)
        x_lm = torch.cat([x_l, x_m], dim=1)
        x_lm = self.conv_fuse_1(x_lm)

        res_lm = x_lm

        x_lm = self.GRSTB_2(x_lm)
        x_lm = self.conv_after_grstb2(x_lm)

        x_lm += res_lm
        x_lm = self.upconv2(x_lm)
        x_lm = self.up(x_lm)
        x_h = torch.cat([x_lm, x_h], dim=1)
        x_h = self.conv_fuse_2(x_h)

        res_h = x_h

        x_h = self.GRSTB_3(x_h)
        x_h = self.conv_after_grstb3(x_h)
        x_h += res_h
        x_h = self.conv_before_up(x_h)
        x_h = self.upsample(x_h)
        x_h = self.conv_last(x_h)        
        x_h += self.mean
        x_h = x_h[:, :, :H*self.scale, :W*self.scale]

        return x_h
