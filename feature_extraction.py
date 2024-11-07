import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict

class FeatureExtractionModule(nn.Module):
    def __init__(self):
        super(FeatureExtractionModule, self).__init__()

        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)

        self.layer1 = nn.Sequential(*list(resnet.children())[:4])
        self.layer2 = resnet.layer1  
        self.layer3 = resnet.layer2   
        self.layer4 = resnet.layer3   
        self.layer5 = resnet.layer4   

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )

    def forward(self, x):
        c1 = self.layer1(x)   
        c2 = self.layer2(c1)  
        c3 = self.layer3(c2)  
        c4 = self.layer4(c3)  
        c5 = self.layer5(c4)  

        feature_maps = OrderedDict()
        feature_maps['layer1'] = c2
        feature_maps['layer2'] = c3
        feature_maps['layer3'] = c4
        feature_maps['layer4'] = c5

        fpn_output = self.fpn(feature_maps)
        return fpn_output
