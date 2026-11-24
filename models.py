import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        return x * channel_att

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled_features = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(pooled_features))
        return x * spatial_att

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class EnhancedAutoencoder(nn.Module):
    def __init__(self, cbam_reduction_ratio=8, cbam_kernel_size=5):
        super(EnhancedAutoencoder, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.cbam_enc1 = CBAM(32, cbam_reduction_ratio, cbam_kernel_size)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.cbam_enc2 = CBAM(64, cbam_reduction_ratio, cbam_kernel_size)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.cbam_enc3 = CBAM(128, cbam_reduction_ratio, cbam_kernel_size)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.cbam_bottleneck = CBAM(256, cbam_reduction_ratio, cbam_kernel_size)
        self.attention = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1), nn.Sigmoid())
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.cbam_dec3 = CBAM(128, cbam_reduction_ratio, cbam_kernel_size)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.cbam_dec2 = CBAM(64, cbam_reduction_ratio, cbam_kernel_size)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.cbam_dec1 = CBAM(32, cbam_reduction_ratio, cbam_kernel_size)
        
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x):
        e1_proc = self.enc1(x)
        e1 = self.cbam_enc1(e1_proc)
        p1 = self.pool1(e1)
        
        e2_proc = self.enc2(p1)
        e2 = self.cbam_enc2(e2_proc)
        p2 = self.pool2(e2)
        
        e3_proc = self.enc3(p2)
        e3 = self.cbam_enc3(e3_proc)
        p3 = self.pool3(e3)
        
        b_proc = self.bottleneck(p3)
        b = self.cbam_bottleneck(b_proc)
        att_map = self.attention(b)
        
        d3_up = self.up3(b)
        d3_cat = torch.cat([d3_up, e3], dim=1)
        d3_proc = self.dec3(d3_cat)
        d3 = self.cbam_dec3(d3_proc)
        
        d2_up = self.up2(d3)
        d2_cat = torch.cat([d2_up, e2], dim=1)
        d2_proc = self.dec2(d2_cat)
        d2 = self.cbam_dec2(d2_proc)
        
        d1_up = self.up1(d2)
        d1_cat = torch.cat([d1_up, e1], dim=1)
        d1_proc = self.dec1(d1_cat)
        d1 = self.cbam_dec1(d1_proc)
        
        reconstruction = torch.sigmoid(self.out_conv(d1))
        return reconstruction, att_map

class DenseNet121TumorClassificationModel(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, cbam_reduction_ratio=16, cbam_kernel_size=7):
        super(DenseNet121TumorClassificationModel, self).__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.densenet121 = models.densenet121(weights=weights)
        original_conv0 = self.densenet121.features.conv0
        self.densenet121.features.conv0 = nn.Conv2d(
            1, original_conv0.out_channels, kernel_size=original_conv0.kernel_size,
            stride=original_conv0.stride, padding=original_conv0.padding, bias=False
        )
        if pretrained:
            with torch.no_grad():
                self.densenet121.features.conv0.weight.copy_(original_conv0.weight.mean(dim=1, keepdim=True))
        num_ftrs = self.densenet121.classifier.in_features
        self.cbam = CBAM(num_ftrs, cbam_reduction_ratio, cbam_kernel_size)
        self.classifier_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.densenet121.classifier = nn.Identity()
        self._initialize_weights(self.classifier_head)
        
    def _initialize_weights(self, classifier_module):
        for m in classifier_module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def freeze_backbone(self):
        """Freeze DenseNet features for head-only warmup (faster + more stable)."""
        for p in self.densenet121.features.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def backbone_params(self):
        return list(self.densenet121.features.parameters()) + list(self.cbam.parameters())

    def head_params(self):
        return list(self.classifier_head.parameters())
                    
    def forward(self, x):
        features = self.cbam(self.densenet121.features(x))
        out = F.adaptive_avg_pool2d(F.relu(features, inplace=True), (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier_head(out)
