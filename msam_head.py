import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg_resize import resize

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class MSAM(nn.Module):
    

    def __init__(self, pool_scale=6, in_channels=64, channels=128):
        super(MSAM, self).__init__()
        self.pool_scale = pool_scale
        self.in_channels = in_channels
        self.channels = channels
        self.pooled_redu_conv = nn.Conv2d(self.in_channels, self.channels, 1, 1, 0, bias=False)
        self.pool_bnrelu = BNPReLU(self.channels)
        
        self.input_redu_conv = nn.Conv2d(128, self.channels, 1, 1, 0, bias=False)
        self.input_bnrelu = BNPReLU(self.channels)
        
        self.global_info = nn.Conv2d(self.channels, self.channels, 1, 1, 0, bias=False)
        self.global_bnrelu = BNPReLU(self.channels)

        self.gla = nn.Conv2d(self.channels, self.pool_scale**2, 1, 1, 0, bias=False)
                

        self.residual_conv = nn.Conv2d(self.channels, self.channels, 1, 1, 0, bias=False)
        #Output may require a fusion layer-1x1 conv
        self.maxpool = nn.MaxPool2d(2, stride=2)
        
    def forward(self, x_high, x_low):
        """Forward function."""
        x_high = self.maxpool(x_high)
        pooled_x = F.adaptive_avg_pool2d(x_high, self.pool_scale)
        # [batch_size, channels, h, w]
        x = self.input_bnrelu(self.input_redu_conv(x_low))
        # [batch_size, channels, pool_scale, pool_scale]
        pooled_x = self.pool_bnrelu(self.pooled_redu_conv(pooled_x))
        batch_size = x_low.size(0)
        # [batch_size, pool_scale * pool_scale, channels]
        pooled_x = pooled_x.view(batch_size, self.channels,
                                 -1).permute(0, 2, 1).contiguous()
        # [batch_size, h * w, pool_scale * pool_scale]
        affinity_matrix = self.gla(x + resize(
            self.global_bnrelu(self.global_info(F.adaptive_avg_pool2d(x, 1))), size=x.shape[2:])
                                   ).permute(0, 2, 3, 1).reshape(
                                       batch_size, -1, self.pool_scale**2)
        affinity_matrix = F.sigmoid(affinity_matrix)
        # [batch_size, h * w, channels]
        z_out = torch.matmul(affinity_matrix, pooled_x)
        # [batch_size, channels, h * w]
        z_out = z_out.permute(0, 2, 1).contiguous()
        # [batch_size, channels, h, w]
        z_out = z_out.view(batch_size, self.channels, x.size(2), x.size(3))
        z_out = self.residual_conv(z_out)
        z_out = F.relu(z_out + x)
        
        return z_out
