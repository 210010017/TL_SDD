import torch
import torch.nn as nn
from feature_extraction import FeatureExtractionModule
from feature_reweighting import FeatureReweightingModule
from distance_metric import DistanceMetricModule

class TL_SDD_Model(nn.Module):
    def __init__(self, num_classes=10):
        super(TL_SDD_Model, self).__init__()
        self.feature_extractor = FeatureExtractionModule()
        self.feature_reweighting = FeatureReweightingModule()
        
        self.classifier = nn.Linear(160000, num_classes)  # this we must adjust after finding correct size, we can find this from commented print statement

    def forward(self, x):
        fpn_features = self.feature_extractor(x)
        reweighted_features = self.feature_reweighting(fpn_features)

        layer4_features = reweighted_features['layer4'].view(x.size(0), -1) 
        # print(f"Flattened layer4_features shape: {layer4_features.shape}")
        
        logits = self.classifier(layer4_features)
        return logits

