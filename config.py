import torch

config = {
    "root_dir": "./GC10-DET",
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs_phase1": 4,
    "num_epochs_phase2": 6,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
