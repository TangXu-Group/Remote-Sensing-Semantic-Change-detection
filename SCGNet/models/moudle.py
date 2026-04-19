import math
import torch
import torch.nn as nn
import torch.nn.functional as F
   
def cosine_similarity(x, y):
    dot_product = torch.sum(x * y, dim=1).unsqueeze(1)
    norm_x = torch.norm(x, p=2, dim=1, keepdim=True)  # L2范数
    norm_y = torch.norm(y, p=2, dim=1, keepdim=True)
    similarity = dot_product / (norm_x * norm_y + 1e-8)
    return similarity
#Semantic-Guided Fusion Module  
class SGF(nn.Module):
    def __init__(self, in_channels):
        super(SGF, self).__init__()
        self.in_channels = in_channels
        self.conv_mask = nn.Conv2d(self.in_channels, 32, kernel_size=1)
    
        self.cat_fusion = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.cat_diff_fusion = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        #cat
        x = torch.cat((x1, x2), dim=1)
        x1_proj = self.conv_mask(x1)
        x2_proj = self.conv_mask(x2)
        mask = (1 - cosine_similarity(x1_proj, x2_proj)) / 2  # 计算余弦相似度生成掩码
        cat_fusion = self.cat_fusion(x * mask)  # 语义引导融合
        #diff
        diff_fusion = torch.abs(x1 - x2)
        #cat_diff
        cat_diff_fusion = self.cat_diff_fusion(torch.cat((cat_fusion, diff_fusion), dim=1))
    
        return cat_diff_fusion        
        
#Semantic consistent enhancement module    
class SCE(nn.Module):
    def __init__(self, dim):
        super(SCE, self).__init__()

        self.dim = dim

        self.classifier = nn.Conv2d(self.dim, 1, kernel_size=1, bias=False)

        self.gate1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.gate2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.out1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.out2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, xc):
        change_prob = torch.sigmoid(self.classifier(xc))
        no_change = F.softmax(torch.cat([1 - change_prob, change_prob], dim=1), dim=1)[:, :1, :, :]
        
        gated_x1 = x1 * no_change
        gated_x2 = x2 * no_change

        gate1 = torch.sigmoid(self.gate1(gated_x1))
        gate2 = torch.sigmoid(self.gate2(gated_x2))

        x1 = self.out1(x1 + gated_x2 * gate1)
        x2 = self.out2(x2 + gated_x1 * gate2)

        return x1, x2

#Multi-scale Change Activation Module
class MSCA(nn.Module):
    def __init__(self, dim=128, pool_rates=[1, 2, 4]):
        super(MSCA, self).__init__()
        self.pool_rates = pool_rates
        
        self.pools = nn.ModuleList([
            nn.Identity() if rate == 1 else nn.AvgPool2d(kernel_size=rate, stride=rate)
            for rate in pool_rates
        ])
        
        self.att_convs = nn.ModuleList([
                nn.Conv2d(dim, 1, kernel_size=1, bias=False)
            for _ in pool_rates
        ])

        # 融合特征的卷积（在每次渐进上采样后使用）
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            )
            for _ in range(len(pool_rates) - 1)
        ])

        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, f, x1, x2):
        identity = f
        diff_map = torch.abs(x1 - x2)

        f_multi = []
        for i, pool in enumerate(self.pools):
            f_pooled = pool(f)
            d_pooled = pool(diff_map)
            attn = torch.sigmoid(self.att_convs[i](d_pooled))
            f_att = f_pooled * attn
            f_multi.append(f_att)

        # 从最小尺度开始，逐级上采样和融合
        fused = f_multi[-1]  # 最小尺度特征
        for i in reversed(range(len(f_multi) - 1)):
            up = F.interpolate(fused, size=f_multi[i].shape[2:], mode='bilinear', align_corners=False)
            fused = self.fusion_convs[i](up + f_multi[i])

        # 残差连接并输出
        output = self.output_conv(fused + identity)
        return output
    

