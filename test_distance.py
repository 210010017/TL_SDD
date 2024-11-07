import torch
from feature_extraction import FeatureExtractionModule
from feature_reweighting import FeatureReweightingModule
from distance_metric import DistanceMetricModule

feature_extractor = FeatureExtractionModule()
feature_reweighter = FeatureReweightingModule()
distance_metric = DistanceMetricModule()

input_tensor = torch.randn(1, 3, 1000, 2048)
reference_tensor = torch.randn(1, 3, 1000, 2048) 

fpn_outputs = feature_extractor(input_tensor)
reweighted_features = feature_reweighter(fpn_outputs)

reference_fpn_outputs = feature_extractor(reference_tensor)
reference_reweighted_features = feature_reweighter(reference_fpn_outputs)

distances = distance_metric(reweighted_features, reference_reweighted_features)

print("Normalized Distances for each layer:")
for layer, distance in distances.items():
    print(f"{layer}: {distance}")
