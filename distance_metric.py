import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class DistanceMetricModule(nn.Module):
    def __init__(self):
        super(DistanceMetricModule, self).__init__()
        
    def forward(self, reweighted_features, reference_features):
        """
        Compute the normalized Euclidean distance between reweighted and reference features.
        
        Args:
            reweighted_features (OrderedDict): Reweighted feature maps from the input.
            reference_features (OrderedDict): Reference feature maps to compare against.
        
        Returns:
            distances (dict): Dictionary of normalized distances for each layer.
        """
        distances = {}
        
        for layer_name in reweighted_features.keys():
            input_feat = reweighted_features[layer_name].view(reweighted_features[layer_name].size(0), -1)
            ref_feat = reference_features[layer_name].view(reference_features[layer_name].size(0), -1)
            
            input_feat = (input_feat - input_feat.mean()) / input_feat.std()
            ref_feat = (ref_feat - ref_feat.mean()) / ref_feat.std()
            
            distance = F.pairwise_distance(input_feat, ref_feat, p=2)
            distances[layer_name] = distance.mean().item()
            
        return distances
