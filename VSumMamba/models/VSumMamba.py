import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.models.PositionalEncoding import (
    FixedPositionalEncoding,
    LearnedPositionalEncoding,
)
import torch
import torch.nn as nn
from torchvision import models
from mamba_ssm.modules.mamba_simple import Mamba
from functools import partial
from timm.models.layers import DropPath, to_2tuple
from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _load_weights
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint
import math

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


__all__ = ['VSumMamba_A']

class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
            use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        layer_idx=None,
        bimamba=True,
        device=None,
        dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

class LayerNorm_conv(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def __init__(self, normalized_shape):
        super().__init__(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor):
        x = x.permute(0,2,3,1)
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))# add ssf
        return ret.type(orig_type).permute(0,3,1,2)

class VSumMamba_A(nn.Module):
    def __init__(
            self,
            img_dim,
            patch_dim,
            out_dim,
            num_channels,
            embedding_dim,
            hidden_dim,
            dropout_rate=0.0,
            attn_dropout_rate=0.0,
            conv_patch_representation=False,
            positional_encoding_type="learned",
            depth=12,
            embed_dim=768,
            drop_path_rate=0.1,
            ssm_cfg=None,
            norm_epsilon=1e-5,
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True,
            residual_in_fp32=True,
            bimamba=True,
            device=None,
            dtype=None,
       

    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(VSumMamba_A, self).__init__()


        assert img_dim % patch_dim == 0
        
 
        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim
        self.out_dim = out_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation
        self.aug_dim = aug_dim
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches + 1
        self.flatten_dim = patch_dim * patch_dim * num_channels
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)
        self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
        self.linear_encoding_aug = nn.Linear(self.aug_dim, embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, out_dim),
            )

        self.mlp_multi = nn.Linear(self.num_patches + 1, self.num_patches)
        self.mlp_single = nn.Linear(self.num_patches, int(self.num_patches / self.num_patches))
        self.mlp_single_forloop = nn.Linear(self.embedding_dim, int(self.embedding_dim))
        self.PSMM_norm = nn.LayerNorm(self.embedding_dim)


        self.to_cls_token = nn.Identity()

        self.fused_add_norm = fused_add_norm
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.residual_in_fp32 = residual_in_fp32
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        self.layerm = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=0,
                    bimamba=False,
                    drop_path=inter_dpr[0],
                    **factory_kwargs,
                )
            ]
        )
        self.mamba_stages = []
        scale_factors = [0.5, 1.0, 2.0]  # feature scales used
        dim1 = 512
        
        for idx, scale in enumerate(scale_factors):
            out_dim = dim1
            out_channels = dim1
            if scale == 4.0:
                layr = [
                    nn.ConvTranspose2d(dim1, dim1 // 2, kernel_size=2, stride=2),
                    LayerNorm_conv(dim1 // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim1 // 2, dim1 // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim1 // 4
            elif scale == 2.0:
                layr = [nn.ConvTranspose2d(dim1, dim1 // 2, kernel_size=2, stride=2)]
                out_dim = dim1 // 2
            elif scale == 1.0:
                layr = []
            elif scale == 0.5:
                layr = [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif scale == 0.25:
                layr = [nn.MaxPool2d(kernel_size=4, stride=4)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layr.extend(
                [
                    nn.Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                    ),
                    LayerNorm_conv(out_channels),
                    nn.GELU(),
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    LayerNorm_conv(out_channels)
                ]
            )
            lays = nn.Sequential(*layr).cuda()
            self.mamba_stages.append(lays)
        half=144
        one=784
        two=3136
        self.linear1 = nn.Linear(half,16)
        self.linear2 = nn.Linear(one,16)
        self.linear3 = nn.Linear(two,16)
        self.norm = nn.LayerNorm(dim1)

        self.weights=nn.Parameter(torch.randn(3))
        
        

    def forward(self, x, inference_params=None):
        #Multi-GA
        bs, t, c, h, w = x.shape  # [40,16,512,7,7]
        x = x.view(bs*t,c,h,w)
        visual_mamba_ms=[]
        for stage in self.mamba_stages:
            visual_mamba_ms.append(stage(x).view(bs, t, c, -1).permute(0, 1, 3, 2))
        v=[]
        #scale-wise
        for i in visual_mamba_ms:
            v.append(i.view(bs,-1,c))

        visual_mamba_ms[0] = self.linear1((v[0]).permute(0,2,1)).permute(0,2,1)
        visual_mamba_ms[1] = self.linear2((v[1]).permute(0,2,1)).permute(0,2,1)
        visual_mamba_ms[2] = self.linear3((v[2]).permute(0,2,1)).permute(0,2,1)
        
        weights = F.softmax(self.weights,dim=0)
        x = (visual_mamba_ms[0]*weights[0] + visual_mamba_ms[1]*weights[1] + visual_mamba_ms[2]*weights[2])
        
       
        x = self.norm(x)
        x = self.linear_encoding(x)
       

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)



        residual = None
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                x, residual = layer(
                    x, residual, inference_params=inference_params,
                    use_checkpoint=True
                )
            else:
                x, residual = layer(
                    x, residual, inference_params=inference_params
                )
        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
        x = fused_add_norm_fn(
            self.drop_path(x),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=self.residual_in_fp32,
        )

        x = x.permute(0, 2, 1)
        x = self.mlp_multi(x)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x_outputs = torch.zeros(self.num_patches, bs, self.out_dim).to(device)
        orix1 = self.mlp_single(x).permute(0, 2, 1)
        
        residual = None
        for i in range(self.num_patches):
            x1 = self.mlp_single_forloop(torch.unsqueeze(x.permute(2, 0, 1)[i], 1)) + orix1
            x1_norm = self.PSMM_norm(x1)
            
            
            for idx1, layer1 in enumerate(self.layerm):
                if self.use_checkpoint and idx1 < self.checkpoint_num:
                    x1_PSMM, residual = layer1(
                        x1_norm, residual, inference_params=inference_params,
                        use_checkpoint=True
                    )
                else:
                    x1_PSMM, residual = layer1(
                        x1_norm, residual, inference_params=inference_params
                    )

            
            x1_PSMM = x1_norm
            x1_PSMM = x1_PSMM + x1
            x1_PSMM = self.pre_head_ln(x1_PSMM)
            x1_PSMM = self.to_cls_token(x1_PSMM[:, 0])
            x1_PSMM = self.mlp_head(x1_PSMM)
            x_outputs[i] = x1_PSMM



        return x_outputs

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)


def VSumMamba_A(dataset='SumMe'):
    
    elif dataset == 'TVSum_77':
        img_dim = 4
        out_dim = 2
        patch_dim = 1
    elif dataset == 'SumMe_77':
        img_dim = 4
        out_dim = 2
        patch_dim = 1
    

    return SpatioTemporal_Vision_Transformer_03(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        # edited
        num_channels=512,
        embedding_dim=768,
        hidden_dim=3072,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,

        conv_patch_representation=False,
        positional_encoding_type="learned",
    )