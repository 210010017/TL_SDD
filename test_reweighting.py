import torch
from feature_extraction import FeatureExtractionModule
from feature_reweighting import FeatureReweightingModule

feature_extractor = FeatureExtractionModule()
feature_reweighter = FeatureReweightingModule()

input_tensor = torch.randn(1, 3, 1000, 2048)

fpn_outputs = feature_extractor(input_tensor)

reweighted_features = feature_reweighter(fpn_outputs)

print("Shapes after reweighting:")
for layer_name, feature_map in reweighted_features.items():
    print(f"{layer_name} shape: {feature_map.shape}")

print("\nSample values before and after reweighting:")
for layer_name in fpn_outputs.keys():
    print(f"\n{layer_name} - Original Mean: {fpn_outputs[layer_name].mean().item()}, Reweighted Mean: {reweighted_features[layer_name].mean().item()}")
