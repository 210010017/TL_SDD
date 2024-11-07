import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class FeatureReweightingModule(nn.Module):
    def __init__(self):
        super(FeatureReweightingModule, self).__init__()
        
        self.channel_attention1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(256, 64, kernel_size=1),  
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=1),  
            nn.Sigmoid()  
        )

        self.channel_attention2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=1),
            nn.Sigmoid()
        )

        self.channel_attention3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=1),
            nn.Sigmoid()
        )

        self.channel_attention4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, fpn_outputs):
        reweighted_features = OrderedDict()
        
        reweighted_features['layer1'] = fpn_outputs['layer1'] * self.channel_attention1(fpn_outputs['layer1'])
        reweighted_features['layer2'] = fpn_outputs['layer2'] * self.channel_attention2(fpn_outputs['layer2'])
        reweighted_features['layer3'] = fpn_outputs['layer3'] * self.channel_attention3(fpn_outputs['layer3'])
        reweighted_features['layer4'] = fpn_outputs['layer4'] * self.channel_attention4(fpn_outputs['layer4'])
        
        return reweighted_features
