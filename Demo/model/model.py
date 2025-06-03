import math
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, DropPath
from timm.models.vision_transformer import PatchEmbed
from model.pos_embed import get_2d_sincos_pos_embed



class PatchEmbed_padding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size_ = patch_size
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1] + 1) * (img_size[0] // patch_size[0] + 1)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=patch_size_//2)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h
        
    def forward(self, x, spatial_size=None):
        B, N, C = x.shape 
        if spatial_size is None:
            a = b = int(math.sqrt(N)) # 8
        else:
            a = int(spatial_size[0])
            b = int(spatial_size[1])
        x = x.view(B, a, b, C) 
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho') 
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C) 
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Block_filter(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x))))) 
        return x


class filter1d(nn.Module):
    def __init__(self, h, w,):
        super().__init__()
        self.weight_h = nn.Parameter(torch.ones(h)) 
        self.weight_w = nn.Parameter(torch.ones(w)) 
        
    def forward(self, x):
        x = torch.fft.rfft(x, dim=3) 
        weight_h = self.weight_h.unsqueeze(0).unsqueeze(1).unsqueeze(3).expand_as(x)
        weight_w = self.weight_w.unsqueeze(0).unsqueeze(1).unsqueeze(2).expand_as(x)
        x = x * weight_h
        x = x * weight_w
        x = torch.fft.irfft(x, dim=3)
        return x

        
class SigGen(nn.Module):
    def __init__(self, args, img_size=224, patch_size=16, in_chans=3,
                embed_dim=1024, depth=24, num_heads=16,
                decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_embed_padding = PatchEmbed_padding(img_size, patch_size, in_chans, embed_dim)
        num_patches_padding = self.patch_embed_padding.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False) 
        self.pos_embed_padding = nn.Parameter(torch.zeros(1, num_patches_padding + 1, embed_dim), requires_grad=False) 
        depth_filter = depth 
        dpr = [x.item() for x in torch.linspace(0, 0., depth_filter)]  
        h = img_size // patch_size
        w = h // 2 + 1
        w_filter = img_size //2 +1 
        self.filter_blocks_pre = filter1d(img_size, w_filter)
        self.filter_blocks = nn.ModuleList([
                    Block_filter(dim=embed_dim, mlp_ratio=mlp_ratio,
                        drop=0., drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w)
                    for i in range(depth_filter)])
        h_padding = (img_size+patch_size)//patch_size
        w_padding = h_padding // 2 + 1
        self.filter_blocks_padding = nn.ModuleList([
                    Block_filter(dim=embed_dim, mlp_ratio=mlp_ratio,
                        drop=0., drop_path=dpr[i], norm_layer=norm_layer, h=h_padding, w=w_padding)
                    for i in range(depth_filter)])
        self.norm = norm_layer(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  
        self.decoder_pos_embed_padding = nn.Parameter(torch.zeros(1, num_patches_padding + 1, decoder_embed_dim), requires_grad=False)  
        self.decoder_filter_blocks = nn.ModuleList([
                    Block_filter(dim=decoder_embed_dim, mlp_ratio=mlp_ratio,
                        drop=0., drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w)
                    for i in range(depth_filter)])
        self.decoder_filter_blocks_padding = nn.ModuleList([
                    Block_filter(dim=decoder_embed_dim, mlp_ratio=mlp_ratio,
                        drop=0., drop_path=dpr[i], norm_layer=norm_layer, h=h_padding, w=w_padding)
                    for i in range(depth_filter)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) 
        self.img_size = img_size
        self.initialize_weights()


    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        pos_embed_padding = get_2d_sincos_pos_embed(self.pos_embed_padding.shape[-1], int(self.patch_embed_padding.num_patches**.5), cls_token=True)
        self.pos_embed_padding.data.copy_(torch.from_numpy(pos_embed_padding).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        decoder_pos_embed_padding = get_2d_sincos_pos_embed(self.decoder_pos_embed_padding.shape[-1], int(self.patch_embed_padding.num_patches**.5), cls_token=True)
        self.decoder_pos_embed_padding.data.copy_(torch.from_numpy(decoder_pos_embed_padding).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x,): 
        x_ = x
        x = self.patch_embed(x) 
        x_padding = self.patch_embed_padding(x_) 
        x = x + self.pos_embed[:, 1:, :] 
        x_padding = x_padding + self.pos_embed_padding[:, 1:, :]
        for blk_filter in self.filter_blocks:
            x = blk_filter(x)
        for blk_filter in self.filter_blocks_padding:
            x_padding = blk_filter(x_padding)        
        x = self.norm(x) 
        x_padding = self.norm(x_padding)
        return x, x_padding

    def decoder(self, x, x_padding):
        x = self.decoder_embed(x) 
        x_padding = self.decoder_embed(x_padding) 
        x = x + self.decoder_pos_embed[:, 1:, :]
        x_padding = x_padding + self.decoder_pos_embed_padding[:, 1:, :]
        for blk_filter in self.decoder_filter_blocks:
            x = blk_filter(x)
        for blk_filter in self.decoder_filter_blocks_padding:
            x_padding = blk_filter(x_padding)   
        x = self.decoder_norm(x)
        x_padding = self.decoder_norm(x_padding)
        x = self.decoder_pred(x)
        x_padding = self.decoder_pred(x_padding)
        x = x[:, :, :] 
        x_padding = x_padding[:, :, :] 
        return x, x_padding

    def depatchify(self, x, img_size=128, patch_size=16):
        p = patch_size
        h = w = img_size // p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, img_size, img_size))
        return imgs
    
    def decoder_combine(self, latent, latent_padding):
        pred, pred_padding = self.decoder(latent, latent_padding)
        imgs = self.depatchify(pred, self.img_size, self.patch_size)
        imgs_padding = self.depatchify(pred_padding, self.img_size+self.patch_size, self.patch_size)
        imgs = (imgs + imgs_padding[:,:,self.patch_size//2:self.img_size+self.patch_size//2, self.patch_size//2:self.img_size+self.patch_size//2])/2
        return imgs
    
    def forward(self, imgs):
        imgs = self.filter_blocks_pre(imgs)
        latent, latent_padding = self.forward_encoder(imgs) 
        imgs = self.decoder_combine(latent, latent_padding)
        return imgs, latent, latent_padding

