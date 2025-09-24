import torch
from torch import nn, einsum
import torch.nn.functional as F
from .blocks import (get_sinusoid_encoding, MaskedConv1D, LayerNorm, THR_layer)
import math
from models.ast_models import ASTModel
import models.objectives
from models import heads as HD
from transformers import RobertaConfig, RobertaModel
from models.bert_model import BertCrossLayer
from models.hntl_vit import ViT_feature_VAE_dec, Time_feature_VAE_dec

kl_loss = nn.KLDivLoss(reduction='batchmean')

bert_config = RobertaConfig(
    vocab_size=50265,
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=256 * 4,
    max_position_embeddings=40,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)
text_transformer = RobertaModel(bert_config)
cosine_similarity = nn.CosineSimilarity(dim=-1)


def compute_rec_with_mask(reconstructed_inputs, original_inputs, mask):
    """
    reconstructed_inputs: [B, T, D]
    original_inputs:      [B, T, D]
    mask:                 [B, 1, T] or [B, T] or [B, T, 1] (bool)

    返回：masked MSE loss
    """
    # 检查维度
    assert reconstructed_inputs.shape == original_inputs.shape, "Input shapes must match"
    assert mask.shape[0] == reconstructed_inputs.shape[0], "Batch size mismatch"
    assert mask.shape[2] == reconstructed_inputs.shape[2], "Time dimension mismatch"

    # 转换 mask 为 float 并广播到 [B, D, T]
    mask = mask.float()
    if mask.shape[1] == 1:
        mask = mask.expand(-1, reconstructed_inputs.size(1), -1)  # [B, D, T]

    # 计算平方差并加权
    diff = (reconstructed_inputs - original_inputs) ** 2
    masked_diff = diff * mask

    # 计算有效均值
    loss = masked_diff.sum() / mask.sum().clamp(min=1.0)
    return loss


class eth_net(nn.Module):  # PTH-Net
    def __init__(self, args,
                 n_in,  # input feature dimension
                 n_embd,  # embedding dimension (after convolution)
                 sgp_mlp_dim,  # the numnber of dim in SSD
                 max_len,  # max sequence length
                 arch=(2, 2, 1, 1, 2),  # (embedding, stem, branch_1, branch_1)
                 scale_factor=2,  # dowsampling rate for the branch,
                 with_ln=False,  # if to attach layernorm after conv
                 path_pdrop=0.0,  # droput rate for drop path
                 downsample_type='max',  # how to downsample feature in FPN
                 thr_size=None,  # size of local window for mha
                 k=None,  # the K in SSD
                 use_pos=False,  # use absolute position embedding
                 num_classes=7
                 ):
        super(eth_net, self).__init__()
        if thr_size is None:
            thr_size = [3, 1, 3, 3]
        if k is None:
            k = [1, 3, 5]
        self.arch = arch
        self.thr_size = thr_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_pos = use_pos

        self.model_dec = Time_feature_VAE_dec(init_weights=True, model_name='vit_base')

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_pos:
            pos_embd_4 = get_sinusoid_encoding(int(self.max_len / 4), n_embd) / (n_embd ** 0.5)
            self.register_buffer("pos_embd_4", pos_embd_4, persistent=False)
            pos_embd_8 = get_sinusoid_encoding(int(self.max_len / 2), n_embd) / (n_embd ** 0.5)
            self.register_buffer("pos_embd_8", pos_embd_8, persistent=False)
            pos_embd = get_sinusoid_encoding(int(self.max_len), n_embd) / (n_embd ** 0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_style = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_in
            else:
                in_channels = n_embd
            self.embd.append(MaskedConv1D(
                in_channels, n_embd, self.thr_size[0],
                stride=1, padding=self.thr_size[0] // 2, bias=(not with_ln)
            )
            )
            self.embd_style.append(MaskedConv1D(
                in_channels, n_embd, self.thr_size[0],
                stride=1, padding=self.thr_size[0] // 2, bias=(not with_ln)
            )
            )
            if with_ln:
                self.embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm.append(nn.Identity())
        self.pos_embed = MaskedConv1D(
            n_embd, n_embd, self.thr_size[0],
            stride=1, padding=self.thr_size[0] // 2, groups=n_embd)
        self.pos_embed_norm = LayerNorm(n_embd)
        self.stem = nn.ModuleList()
        self.stem_style = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                THR_layer(n_embd, self.thr_size[1], n_hidden=sgp_mlp_dim, k=k))
            self.stem_style.append(
                THR_layer(n_embd, self.thr_size[1], n_hidden=sgp_mlp_dim, k=k))

        # main branch using transformer with pooling
        self.branch_1 = nn.ModuleList()
        self.branch_1_style = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch_1.append(THR_layer(n_embd, self.thr_size[2], path_pdrop=path_pdrop,
                                           n_hidden=sgp_mlp_dim, k=k))
            self.branch_1_style.append(THR_layer(n_embd, self.thr_size[2], path_pdrop=path_pdrop,
                                                 n_hidden=sgp_mlp_dim, k=k))
        self.branch_2 = nn.ModuleList()
        self.branch_2_style = nn.ModuleList()
        for idx in range(arch[3]):
            self.branch_2.append(THR_layer(n_embd, self.thr_size[3], path_pdrop=path_pdrop,
                                           n_hidden=sgp_mlp_dim, k=k))
            self.branch_2_style.append(THR_layer(n_embd, self.thr_size[3], path_pdrop=path_pdrop,
                                                 n_hidden=sgp_mlp_dim, k=k))

        self.branch_3 = nn.ModuleList()
        for idx in range(arch[4]):
            self.branch_3.append(THR_layer(n_embd, self.thr_size[4], path_pdrop=path_pdrop,
                                           n_hidden=sgp_mlp_dim, k=k))

        self.head = nn.Linear(n_embd, num_classes) if num_classes > 0 else nn.Identity()
        self.head_audio = nn.Linear(n_embd, num_classes) if num_classes > 0 else nn.Identity()
        self.head_style = nn.Linear(n_embd, num_classes) if num_classes > 0 else nn.Identity()

        self.audio_model = ASTModel(label_dim=args.num_class, fstride=args.fstride, tstride=args.tstride,
                                    input_fdim=128,
                                    input_tdim=args.audio_length, imagenet_pretrain=args.imagenet_pretrain,
                                    audioset_pretrain=False, model_size='base384')

        self.bert_config = RobertaConfig(
            vocab_size=50265,
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=256 * 4,
            max_position_embeddings=40,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.cross_modal_audio_transform = nn.Linear(768, 512)
        self.cross_modal_audio_transform.apply(models.objectives.init_weights)
        self.cross_modal_image_transform = nn.Linear(512, 512)
        self.cross_modal_image_transform.apply(models.objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(6, 512)
        num_fusion = 6
        self.token_type_embeddings.apply(models.objectives.init_weights)
        self.cross_modal_image_layers = nn.ModuleList(
            [BertCrossLayer(self.bert_config) for _ in range(num_fusion)])
        self.cross_modal_image_layers.apply(models.objectives.init_weights)
        self.cross_modal_audio_layers = nn.ModuleList(
            [BertCrossLayer(self.bert_config) for _ in range(num_fusion)])
        self.cross_modal_audio_layers.apply(models.objectives.init_weights)

        self.cross_modal_image_pooler = HD.Pooler(512)
        self.cross_modal_image_pooler.apply(models.objectives.init_weights)
        self.cross_modal_audio_pooler = HD.Pooler(512)
        self.cross_modal_audio_pooler.apply(models.objectives.init_weights)
        temp = 300  # 144
        # self.mapping_H = nn.Linear(temp, 16)
        # self.mapping_M = nn.Linear(temp, 8)
        # self.mapping_L = nn.Linear(temp, 4)

        self.mapping_H = nn.Linear(temp, 4)
        self.mapping_M = nn.Linear(temp, 2)
        self.mapping_L = nn.Linear(temp, 1)

        self.ln = LayerNorm(512)

        if self.scale_factor > 1:
            if downsample_type == 'max':
                kernel_size, stride, padding = \
                    self.scale_factor + 1, self.scale_factor, (self.scale_factor + 1) // 2
                self.downsample = nn.MaxPool1d(
                    kernel_size, stride=stride, padding=padding)
                self.stride = stride
            elif downsample_type == 'avg':
                self.downsample = nn.Sequential(nn.AvgPool1d(self.scale_factor, stride=self.scale_factor, padding=0),
                                                nn.Conv1d(n_embd, n_embd, 1, 1, 0))
                self.stride = self.scale_factor
            else:
                raise NotImplementedError("downsample type error")
        else:
            self.downsample = nn.Identity()
            self.stride = 1

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward_features(self, x, mask, training=True, audio=None):
        x_style = x
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))
            x_style, _ = self.embd_style[idx](x_style, mask)
            x_style = self.relu(self.embd_norm[idx](x_style))

        B, C, T = x.shape

        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](None, None, x, None, None, mask, False)
            x_style, _ = self.stem_style[idx](None, None, x_style, None, None, mask, False)

        if self.arch[2] == 0:
            return x, mask

        x = self.downsample(x)
        x_style = self.downsample(x_style)
        out_mask = F.interpolate(
            mask.to(x.dtype),
            size=torch.div(T, self.stride, rounding_mode='trunc'),
            mode='nearest'
        ).detach()
        mask = out_mask
        # main branch with downsampling
        for idx in range(len(self.branch_1)):
            x, mask = self.branch_1[idx](None, None, x, None, None, mask, False)
            x_style, _ = self.branch_1[idx](None, None, x_style, None, None, mask, False)

        if self.arch[3] == 0:
            return x, mask

        B, C, T = x.shape
        x = self.downsample(x)
        x_style = self.downsample(x_style)
        out_mask = F.interpolate(
            mask.to(x.dtype),
            size=torch.div(T, self.stride, rounding_mode='trunc'),
            mode='nearest'
        ).detach()
        mask = out_mask

        for idx in range(len(self.branch_2)):
            x, mask = self.branch_2[idx](None, None, x, None, None, mask, False)
            x_style, _ = self.branch_2_style[idx](None, None, x_style, None, None, mask, False)
        if training:
            x_rec, out_mask = self.model_dec(x, x_style, mask, mask)

        x = x.permute(0, 2, 1)
        x = self.cross_modal_image_transform(x)
        video_masks = mask[:, 0, :]
        # 将bool 转为0，1
        video_masks = video_masks.to(dtype=torch.long)
        extend_video_masks = text_transformer.get_extended_attention_mask(attention_mask=video_masks,
                                                                          input_shape=video_masks.size(),
                                                                          device=x.device)
        """add audio data"""
        audio_embeds = self.audio_model(audio)
        audio_embeds = self.cross_modal_audio_transform(audio_embeds)
        audio_embeds = audio_embeds[:, 2:]
        audio_embeds = self.mapping_L(audio_embeds.permute(0, 2, 1))
        audio_embeds = self.ln(audio_embeds)
        audio_embeds = audio_embeds.transpose(1, 2)
        audio_masks = torch.ones((audio_embeds.size(0), audio_embeds.size(1)), dtype=torch.long,
                                 device=audio_embeds.device)
        extend_audio_masks = text_transformer.get_extended_attention_mask(attention_mask=audio_masks,
                                                                          input_shape=audio_masks.size(),
                                                                          device=audio_embeds.device)

        y, x = (
            audio_embeds + self.token_type_embeddings(torch.full_like(audio_masks, 1)),
            x + self.token_type_embeddings(torch.zeros_like(video_masks)),
        )
        if training:
            kl_H = kl_loss(F.log_softmax(y, dim=-1), F.softmax(x, dim=-1))
        # X, Y = image_embeds, audio_embeds
        for image_layer, audio_layer in zip(self.cross_modal_image_layers, self.cross_modal_audio_layers):
            y1 = audio_layer(y, x, extend_audio_masks, extend_video_masks)
            x1 = image_layer(x, y, extend_video_masks, extend_audio_masks)
            x, y = x1[0], y1[0]
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)

        for idx in range(len(self.branch_3)):
            x, mask = self.branch_3[idx](None, None, x, None, None, mask, False)

        if training:
            return x, mask, y, kl_H, x_rec
        else:
            return x, mask, y

        # if training:
        #     return x, mask, None, 0, x_rec
        # else:
        #     return x, mask, None

    def NTL(self, x_4, x_8, x, mask_4, mask_8, mask, training=True):
        if training:
            x_style = x
            x_style_4 = x_4
            x_style_8 = x_8
            for idx in range(len(self.embd)):
                x_4, mask_4 = self.embd[idx](x_4, mask_4)
                x_4 = self.relu(self.embd_norm[idx](x_4))
                x_8, mask_8 = self.embd[idx](x_8, mask_8)
                x_8 = self.relu(self.embd_norm[idx](x_8))
                x, mask = self.embd[idx](x, mask)
                x = self.relu(self.embd_norm[idx](x))
                x_style, _ = self.embd_style[idx](x_style, mask)
                x_style = self.relu(self.embd_norm[idx](x_style))
                x_style_4, _ = self.embd_style[idx](x_style_4, mask_4)
                x_style_4 = self.relu(self.embd_norm[idx](x_style_4))
                x_style_8, _ = self.embd_style[idx](x_style_8, mask_8)
                x_style_8 = self.relu(self.embd_norm[idx](x_style_8))

            for idx in range(len(self.stem)):
                x_4, x_8, x, mask_4, mask_8, mask = self.stem[idx](x_4, x_8, x, mask_4, mask_8, mask, True)
                x_style_4, x_style_8, x_style, _, _, _ = self.stem_style[idx](x_style_4, x_style_8, x_style, mask_4,
                                                                              mask_8, mask, True)

            B, C, T = x_4.shape
            x = self.downsample(x)
            x_style = self.downsample(x_style)
            out_mask = F.interpolate(
                mask.to(x.dtype),
                size=torch.div(T * 4, self.stride, rounding_mode='trunc'),
                mode='nearest'
            ).detach()
            x_4 = self.downsample(x_4)
            x_style_4 = self.downsample(x_style_4)
            out_mask_4 = F.interpolate(
                mask_4.to(x_4.dtype),
                size=torch.div(T, self.stride, rounding_mode='trunc'),
                mode='nearest'
            ).detach()
            x_8 = self.downsample(x_8)
            x_style_8 = self.downsample(x_style_8)
            out_mask_8 = F.interpolate(
                mask_8.to(x_8.dtype),
                size=torch.div(T * 2, self.stride, rounding_mode='trunc'),
                mode='nearest'
            ).detach()
            mask_4 = out_mask_4
            mask_8 = out_mask_8
            mask = out_mask

            for idx in range(len(self.branch_1)):
                x_4, x_8, x, mask_4, mask_8, mask = self.branch_1[idx](x_4, x_8, x, mask_4, mask_8, mask, True)
                x_style_4, x_style_8, x_style, _, _, _ = self.branch_1_style[idx](x_style_4, x_style_8, x_style, mask_4,
                                                                                  mask_8, mask, True)

            B, C, T = x_4.shape
            x = self.downsample(x)
            x_style = self.downsample(x_style)
            out_mask = F.interpolate(
                mask.to(x.dtype),
                size=torch.div(T * 4, self.stride, rounding_mode='trunc'),
                mode='nearest'
            ).detach()
            x_4 = self.downsample(x_4)
            x_style_4 = self.downsample(x_style_4)
            out_mask_4 = F.interpolate(
                mask_4.to(x_4.dtype),
                size=torch.div(T, self.stride, rounding_mode='trunc'),
                mode='nearest'
            ).detach()
            x_8 = self.downsample(x_8)
            x_style_8 = self.downsample(x_style_8)
            out_mask_8 = F.interpolate(
                mask_8.to(x_8.dtype),
                size=torch.div(T * 2, self.stride, rounding_mode='trunc'),
                mode='nearest'
            ).detach()
            mask_4 = out_mask_4
            mask_8 = out_mask_8
            mask = out_mask

            for idx in range(len(self.branch_2)):
                x_4, x_8, x, mask_4, mask_8, mask = self.branch_2[idx](x_4, x_8, x, mask_4, mask_8, mask, True)
                x_style_4, x_style_8, x_style, _, _, _ = self.branch_2_style[idx](x_style_4, x_style_8, x_style, mask_4,
                                                                                  mask_8, mask, True)
            x_rec, out_mask = self.model_dec(x, x_style, mask, mask)
            x_rec_4, out_mask_4 = self.model_dec(x_4, x_style_4, mask_4, mask_4)
            x_rec_8, out_mask_8 = self.model_dec(x_8, x_style_8, mask_8, mask_8)
            return x_4, x_8, x, mask_4, mask_8, mask, x_rec_4, x_rec_8, x_rec
        else:
            for idx in range(len(self.embd)):
                x, mask = self.embd[idx](x, mask)
                x = self.relu(self.embd_norm[idx](x))

            for idx in range(len(self.stem)):
                x, mask = self.stem[idx](None, None, x, None, None, mask, False)

            B, C, T = x.shape
            x = self.downsample(x)
            out_mask = F.interpolate(
                mask.to(x.dtype),
                size=torch.div(T, self.stride, rounding_mode='trunc'),
                mode='nearest'
            ).detach()
            mask = out_mask

            for idx in range(len(self.branch_1)):
                x, mask = self.branch_1[idx](None, None, x, None, None, mask, False)

            B, C, T = x.shape
            x = self.downsample(x)
            out_mask = F.interpolate(
                mask.to(x.dtype),
                size=torch.div(T, self.stride, rounding_mode='trunc'),
                mode='nearest'
            ).detach()
            mask = out_mask

            for idx in range(len(self.branch_2)):
                x, mask = self.branch_2[idx](None, None, x, None, None, mask, False)
            return x, mask

    def confusion(self, x, mask_x, y, mask_y, training=True):
        x = x.permute(0, 2, 1)
        x = self.cross_modal_image_transform(x)
        video_masks = mask_x[:, 0, :]
        # 将bool 转为0，1
        video_masks = video_masks.to(dtype=torch.long)
        extend_video_masks = text_transformer.get_extended_attention_mask(attention_mask=video_masks,
                                                                          input_shape=video_masks.size(),
                                                                          device=x.device)
        extend_audio_masks = text_transformer.get_extended_attention_mask(attention_mask=mask_y,
                                                                          input_shape=mask_y.size(),
                                                                          device=mask_y.device)

        y, x = (
            y + self.token_type_embeddings(torch.full_like(mask_y, 1)),
            x + self.token_type_embeddings(torch.zeros_like(video_masks)),
        )
        if training:
            kl = kl_loss(F.log_softmax(y, dim=-1), F.softmax(x, dim=-1))
        # X, Y = image_embeds, audio_embeds
        for image_layer, audio_layer in zip(self.cross_modal_image_layers, self.cross_modal_audio_layers):
            y1 = audio_layer(y, x, extend_audio_masks, extend_video_masks)
            x1 = image_layer(x, y, extend_video_masks, extend_audio_masks)
            x, y = x1[0], y1[0]
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)
        if training:
            return x, y, kl
        return x, y

    def forward_features_LRM(self, x_4, x_8, x, mask_4, mask_8, mask, training=True, audio=None):
        """add audio data"""
        audio_embeds = self.audio_model(audio)
        audio_embeds = self.cross_modal_audio_transform(audio_embeds)
        audio_embeds = audio_embeds[:, 2:]

        if training:
            x_4, x_8, x, out_mask_4, out_mask_8, out_mask, x_rec_4, x_rec_8, x_rec = self.NTL(x_4, x_8, x, mask_4,
                                                                                              mask_8, mask, training)
            audio_embeds_16 = self.mapping_H(audio_embeds.permute(0, 2, 1))
            audio_embeds_16 = self.ln(audio_embeds_16)
            audio_embeds_16 = audio_embeds_16.transpose(1, 2)
            audio_masks_16 = torch.ones((audio_embeds_16.size(0), audio_embeds_16.size(1)), dtype=torch.long,
                                        device=audio_embeds_16.device)
            audio_embeds_4 = self.mapping_L(audio_embeds.permute(0, 2, 1))
            audio_embeds_4 = self.ln(audio_embeds_4)
            audio_embeds_4 = audio_embeds_4.transpose(1, 2)
            audio_masks_4 = torch.ones((audio_embeds_4.size(0), audio_embeds_4.size(1)), dtype=torch.long,
                                       device=audio_embeds_4.device)
            audio_embeds_8 = self.mapping_M(audio_embeds.permute(0, 2, 1))
            audio_embeds_8 = self.ln(audio_embeds_8)
            audio_embeds_8 = audio_embeds_8.transpose(1, 2)
            audio_masks_8 = torch.ones((audio_embeds_8.size(0), audio_embeds_8.size(1)), dtype=torch.long,
                                       device=audio_embeds_8.device)

            x, y, kl = self.confusion(x, out_mask, audio_embeds_16, audio_masks_16, True)
            x_4, y_4, kl_4 = self.confusion(x_4, out_mask_4, audio_embeds_4, audio_masks_4, True)
            x_8, y_8, kl_8 = self.confusion(x_8, out_mask_8, audio_embeds_8, audio_masks_8, True)
            for idx in range(len(self.branch_3)):
                x_4, x_8, x, mask_4, mask_8, mask = self.branch_3[idx](x_4, x_8, x, out_mask_4, out_mask_8, out_mask, True)

            return x_4, x_8, x, mask_4, mask_8, mask, y, x_rec_4, x_rec_8, x_rec, kl
        else:
            audio_embeds = self.mapping_H(audio_embeds.permute(0, 2, 1))
            audio_embeds = self.ln(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)
            audio_masks = torch.ones((audio_embeds.size(0), audio_embeds.size(1)), dtype=torch.long,
                                        device=audio_embeds.device)
            x, mask = self.NTL(None, None, x, None, None, mask, False)
            x, y = self.confusion(x, mask, audio_embeds, audio_masks, False)
            for idx in range(len(self.branch_3)):
                x, mask = self.branch_3[idx](None, None, x, None, mask_8, mask, False)
            return x, mask, y

    def forward(self, x_4, x_8, x, mask_4, mask_8, mask, training=True, audio=None):
        mode = ['single_modality', 'multi_modality'][1]
        a = 0.2
        if mode == 'single_modality':
            if training:
                x, mask, y, kl, x_rec = self.forward_features(x, mask, True, audio)
                x = x.mean(dim=2)
                x = self.head(x)
                y = y.mean(dim=2)
                y = self.head_audio(y)
                x = x * (1 - a) + y * a
                return x, kl, x_rec
            else:
                if x_4 is not None:
                    x, mask = self.forward_features(x_4, None, None, mask_4, None, None, False, audio)
                    x = x * mask
                    x = x.mean(dim=2)
                    # x = x.max(dim=2).values
                    x = self.head(x)
                elif x_8 is not None:
                    x, mask = self.forward_features(None, x_8, None, None, mask_8, None, False, audio)
                    x = x * mask
                    x = x.mean(dim=2)
                    x = self.head(x)
                else:
                    x, mask, y = self.forward_features(x, mask, False, audio)
                    x = x.mean(dim=2)
                    x = self.head(x)
                    y = y.mean(dim=2)
                    y = self.head_audio(y)
                    x = x * (1 - a) + y * a
                return x
        else:
            if training:
                output_x_4, output_x_8, output_x, output_mask_4, output_mask_8, output_mask, y, x_rec_4, x_rec_8, x_rec, kl = self.forward_features_LRM(
                    x_4, x_8, x, mask_4, mask_8, mask, True, audio)
                output_x_4 = output_x_4 * output_mask_4
                output_x_8 = output_x_8 * output_mask_8
                output_x = output_x * output_mask
                output_x_4 = output_x_4.mean(dim=2)
                output_x_8 = output_x_8.mean(dim=2)
                output_x = output_x.mean(dim=2)
                cross_L = 0.5 + torch.mean(cosine_similarity(output_x, output_x_4)) / 2
                cross_M = 0.5 + torch.mean(cosine_similarity(output_x, output_x_8)) / 2
                output_x = self.head(output_x)
                y = y.mean(dim=2)
                y = self.head_audio(y)
                output_x = output_x * (1 - a) + y * a
                loss_rec_4 = compute_rec_with_mask(x_rec_4, x_4, mask_4)
                loss_rec_8 = compute_rec_with_mask(x_rec_8, x_8, mask_8)
                loss_rec = compute_rec_with_mask(x_rec, x, mask)
                return output_x, kl, cross_L + cross_M, loss_rec_4 + loss_rec_8 + loss_rec
            else:
                x, mask, y = self.forward_features_LRM(None, None, x, None, None, mask, False, audio)
                x = x.mean(dim=2)
                x = self.head(x)
                y = y.mean(dim=2)
                y = self.head_audio(y)
                x = x * (1 - a) + y * a
                return x


def PTH_Net():
    max_len = 16
    k = [1, 3, 5]
    thr_size = [3, 1, 3, 3, 3]
    arch = (2, 2, 1, 1, 2)
    return eth_net(n_in=1408, n_embd=512, mlp_dim=768, max_len=max_len, arch=arch,
                   scale_factor=2, with_ln=True, path_pdrop=0.1, downsample_type='max',
                   thr_size=thr_size,
                   k=k, init_conv_vars=0, use_pos=False)

