"""
Import ViT models from timm library and modify them to return intermediate features
timm lid: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
"""

import torch
import torch.nn as nn
import timm
import math
import torch.nn.functional as F

__all__ = [
    'ViT', 'vit_tiny', 'vit_base', 'vit_large', 'vit_huge'
]


class ViT(nn.Module):
    def __init__(self, model_name, pretrained=True, **kwargs):
        super(ViT, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, **kwargs)

        self.patch_embed = self.model.patch_embed
        self._pos_embed = self.model._pos_embed
        self.patch_drop = self.model.patch_drop
        self.norm_pre = self.model.norm_pre
        self.blocks = self.model.blocks
        self.norm = self.model.norm
        self.pool = self.model.pool
        self.fc_norm = self.model.fc_norm
        self.head = self.model.head

        self.forward_head = self.model.forward_head

    def forward(self, x, y=None):
        """
        Get final output and middle block features
        """
        if y == None:
            return self.model(x)
        else:
            # get all layer features
            feature_x_f, feature_layer_x = self.model.forward_intermediate(x)
            feature_y_f, feature_layer_y = self.model.forward_intermediate(y)

            # get middle block features
            split = len(feature_layer_x) // 2
            feature_x = feature_layer_x[split]
            feature_y = feature_layer_y[split]

            # get final output
            out_x = self.model.forward_head(feature_x_f)
            out_y = self.model.forward_head(feature_y_f)

            return out_x, out_y, self.model.pool(feature_x), self.model.pool(feature_y)

    def forward_f(self, x):
        """
        Get final output and final block features
        """
        # get final block features
        feature_f = self.model.forward_features(x)

        # get final output
        out = self.model.forward_head(feature_f)

        return out, self.model.pool(feature_f)

    def forward_layer_f(self, x):
        """
        Get final output and all block features
        """
        # get all block features
        feature_f, layer_features = self.model.forward_intermediates(x)

        # Process layer features via pooling
        layer_features = [self.model.pool(layer) for layer in layer_features]

        # get final output
        out = self.forward_head(feature_f)

        return out, layer_features

    def init_weights(self, mode=''):
        self.model.init_weights(mode)


class ViT_feature_VAE_enc(nn.Module):
    def __init__(self, config=None):
        super(ViT_feature_VAE_enc, self).__init__()

        self.init_weights = not config.teacher_pretrain
        model_name = {
            'vit_tiny': 'vit_tiny_patch16_224',
            'vit_base': 'vit_base_patch16_224',
            'vit_large': 'vit_large_patch16_224',
            'vit_huge': 'vit_huge_patch14_224',
        }
        self.model_name = model_name[config.teacher_network]
        self.model = timm.create_model(
            self.model_name,
            pretrained=config.teacher_pretrain,
            num_classes=config.num_classes,
            img_size=config.image_size).to(config.device)

        # feature extractor
        self.patch_embed = self.model.patch_embed
        self._pos_embed = self.model._pos_embed
        self.patch_drop = self.model.patch_drop
        self.norm_pre = self.model.norm_pre
        self.blocks = self.model.blocks
        self.norm = self.model.norm

        # vae
        f_dim = {
            'vit_tiny': 192,
            'vit_base': 768,
            'vit_large': 1024,
            'vit_huge': 1280,
        }
        self.f_dim = f_dim[config.teacher_network]
        self.latent_dim = int(self.f_dim / 2)
        self.fc_mu = nn.Linear(self.f_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.f_dim, self.latent_dim)

        # classifier
        self.pool = self.model.pool
        # self.fc_norm = self.model.fc_norm
        # self.head_drop = self.model.head_drop
        # self.head = self.model.head
        self.fc_norm = nn.LayerNorm(self.latent_dim)
        self.head_drop = nn.Dropout(0.1)
        self.head = nn.Linear(self.latent_dim, config.num_classes)

        if self.init_weights:
            self.initialize_weights()

    def forward(self, x):
        # feature extraction
        f = self.model.forward_features(x)
        f = self.pool(f)

        # vae
        f_mu, f_logvar = self.fc_mu(f), self.fc_logvar(f)
        f_rp = self.reparameterize(f_mu, f_logvar)

        # classifier
        pred = self.fc_norm(f_rp)
        pred = self.head_drop(pred)
        pred = self.head(pred)

        # x_rec = self.decoder(f_rp)
        if self.training:
            outputs = {
                'f_mu': f_mu,
                'f_logvar': f_logvar,
                'f_rp': f_rp,
                'pred': pred,
                # 'x_rec': x_rec,
            }
        else:
            outputs = pred
        return outputs

    def forward_full(self, x):
        # feature extraction
        f = self.model.forward_features(x)
        f = self.pool(f)

        # vae
        f_mu, f_logvar = self.fc_mu(f), self.fc_logvar(f)
        f_rp = self.reparameterize(f_mu, f_logvar)

        # classifier
        pred = self.fc_norm(f_rp)
        pred = self.head_drop(pred)
        pred = self.head(pred)

        # x_rec = self.decoder(f_rp)
        outputs = {
            'f_mu': f_mu,
            'f_logvar': f_logvar,
            'f_rp': f_rp,
            'pred': pred,
            # 'x_rec': x_rec,
        }
        return outputs

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def initialize_weights(self):
        self.model.init_weights()

    def disentangle(self, f_rp, reverse=False):
        if reverse:
            raise Exception
            # using gradiant reveral layer: use entropy minimization loss
            # else, use negative entropy
            x = self.grl(x)
        with torch.no_grad():
            pred = self.fc_norm(f_rp)
            pred = self.head_drop(pred)
            pred = self.head(pred)
        return pred

    def pred(self, f_rp):
        with torch.no_grad():
            pred = self.fc_norm(f_rp)
            pred = self.head_drop(pred)
            pred = self.head(pred)
        return pred


class Data_Decoder_CIFAR(nn.Module):
    def __init__(self, hidden_dims=[256, 128, 64, 32], z_dim=1):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.decoder_input = nn.Linear(z_dim, hidden_dims[0] * 4)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z, mask=None):
        if mask is not None:
            # 自动对 mask 做 repeat 扩展（如果 channel 数不匹配）
            if mask.shape[1] != z.shape[1]:
                repeat_factor = z.shape[1] // mask.shape[1]
                mask = mask.repeat(1, repeat_factor, 1)
            z = z * mask  # 应用特征 mask
        B, C, T = z.shape
        z = z.permute(0, 2, 1).reshape(B * T, C)  # [B*T, C]
        out = self.decoder_input(z)
        out = out.view(-1, self.hidden_dims[0], 2, 2)
        out = self.decoder(out)
        out = self.final_layer(out)
        return out.view(B, T, 3, 32, 32)


class ViT_feature_VAE_dec(nn.Module):
    def __init__(self, model_name='vit_base', init_weights=True):
        super(ViT_feature_VAE_dec, self).__init__()
        # Get dataset specific configurations

        # Setup the decoder config
        if model_name == 'vit_tiny':
            self.f_dim = 192
            self.hidden_dims = [192, 128, 64, 32]
        elif model_name == 'vit_base':
            self.f_dim = 768
            self.hidden_dims = [768, 256, 128, 64, 32]
        elif model_name == 'vit_large':
            self.f_dim = 1024
            self.hidden_dims = [1024, 256, 128, 64, 32]
        elif model_name == 'vit_huge':
            self.f_dim = 1280
            self.hidden_dims = [1280, 256, 128, 64, 32]
        elif model_name == 'PTH-Net':
            self.f_dim = 512
            self.hidden_dims = [512, 256, 128, 256, 768]
        else:
            raise ValueError(f'Unsupported model name: {model_name}')

        self.decoder = Data_Decoder_CIFAR(hidden_dims=self.hidden_dims, z_dim=self.f_dim)

        if init_weights:
            self.initialize_weights()

    def forward(self, f_rp_c, f_rp_s, mask_c, mask_s):
        f_rp = torch.cat([f_rp_c, f_rp_s], dim=-1)
        f_mask = torch.cat([mask_c, mask_s], dim=-1)
        x_rec = self.decoder(f_rp, f_mask)
        return x_rec

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. 支持mask传播。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding), \
            "kernel_size 必须为奇数，且 padding = kernel_size // 2"
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        if bias:
            nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: [B, C, T], mask: [B, 1, T] (bool)
        B, C, T = x.size()
        assert T % self.stride == 0
        out_conv = self.conv(x)
        if self.stride > 1:
            out_mask = F.interpolate(mask.float(), size=out_conv.size(-1), mode='nearest')
        else:
            out_mask = mask.float()
        out_conv = out_conv * out_mask.detach()
        return out_conv, out_mask.bool()


class MaskedTimeDecoder(nn.Module):
    """
    基于时间维度的解码器，带掩码传播。目标输出为 [B, 16, 756]。
    """

    def __init__(self, hidden_dims=[512, 512, 768], z_dim=512, out_dim=768):
        super(MaskedTimeDecoder, self).__init__()

        self.input_fc = nn.Linear(z_dim, hidden_dims[0])

        self.decoder_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.decoder_layers.append(nn.ModuleDict({
                'upsample': nn.Upsample(scale_factor=2, mode='nearest'),
                'masked_conv': MaskedConv1D(
                    in_channels=hidden_dims[i],
                    out_channels=hidden_dims[i + 1],
                    kernel_size=3,
                    padding=1
                ),
                'norm': nn.BatchNorm1d(hidden_dims[i + 1]),
                'act': nn.LeakyReLU()
            }))

        self.output_layer = MaskedConv1D(
            in_channels=hidden_dims[-1],
            out_channels=out_dim,
            kernel_size=3,
            padding=1
        )

    def forward(self, z, mask):
        """
        z: [B, 4, 512], mask: [B, 1, 4]
        输出: [B, 16, 756], new_mask: [B, 1, 16]
        """
        z = z.permute(0, 2, 1)
        B, T, D = z.size()
        out = self.input_fc(z)  # [B, 4, C]
        out = out.permute(0, 2, 1)  # [B, C, T]
        out_mask = mask  # [B, 1, T]

        for layer in self.decoder_layers:
            out = layer['upsample'](out)  # 时间上采样
            out_mask = F.interpolate(out_mask.float(), scale_factor=2, mode='nearest').bool()
            out, out_mask = layer['masked_conv'](out, out_mask)  # MaskedConv1D
            out = layer['norm'](out)
            out = layer['act'](out)

        out, out_mask = self.output_layer(out, out_mask)  # 最终输出层
        # out = out.permute(0, 2, 1)  # [B, T, D]
        return out, out_mask


class Time_feature_VAE_dec(nn.Module):
    def __init__(self, model_name='vit_base', init_weights=True):
        super(Time_feature_VAE_dec, self).__init__()

        if model_name == 'vit_tiny':
            self.f_dim = 192
        elif model_name == 'vit_base':
            self.f_dim = 1408
        elif model_name == 'vit_large':
            self.f_dim = 1024
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # 假设编码后的维度是 f_dim // 2 + f_dim // 2 = f_dim（拼接内容与风格）
        self.decoder = MaskedTimeDecoder(
            hidden_dims=[512, self.f_dim],  # 2次上采样：4 → 8 → 16
            z_dim=512,
            out_dim=self.f_dim
        )

        if init_weights:
            self.initialize_weights()

    def forward(self, f_rp_c, f_rp_s, mask_c, mask_s):
        """
        f_rp_c, f_rp_s: [B, 2, 256] 各是两个不同来源的时间序列特征
        拼接后：[B, 2, 512] → 解码 → [B, 16, 756]
        """
        f_rp = torch.cat([f_rp_c, f_rp_s], dim=-1)
        f_mask = torch.cat([mask_c, mask_s], dim=-1)
        x_rec, out_mask = self.decoder(f_rp, f_mask)
        return x_rec, out_mask

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def creat_disentangle_model(config, only_student=False, num_styles=2):
    model_dec = ViT_feature_VAE_dec(init_weights=True, config=config)
    model_c = ViT_feature_VAE_enc(config)
    model_s = ViT_feature_VAE_enc(config)

    return model_c, model_s, model_dec


def vit_tiny(pretrained=True, **kwargs):
    """
    patch_size=16, embed_dim=192, depth=12, num_heads=3
    """
    model = ViT('vit_tiny_patch16_224', pretrained, **kwargs)

    return model


def vit_base(pretrained=True, **kwargs):
    """
    patch_size=16, embed_dim=768, depth=12, num_heads=12
    """
    model = ViT('vit_base_patch16_224', pretrained, **kwargs)

    return model


def vit_large(pretrained=True, **kwargs):
    """
    patch_size=16, embed_dim=1024, depth=24, num_heads=16
    """
    model = ViT('vit_large_patch16_224', pretrained, **kwargs)

    return model


def vit_huge(pretrained=True, **kwargs):
    """
    patch_size=14, embed_dim=1280, depth=32, num_heads=16
    
    """
    model = ViT('vit_huge_patch14_224', pretrained, **kwargs)

    return model
