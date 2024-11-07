import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import get_data_loader
from tl_sdd_model import TL_SDD_Model
from config import config
from tqdm import tqdm 

def train_phase(model, data_loader, optimizer, criterion, phase_name):
    model.train()
    running_loss = 0.0
    total, correct = 0, 0
    
    for images, labels in tqdm(data_loader, desc=phase_name):
        images, labels = images.to(config["device"]), labels.to(config["device"])

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    print(f'{phase_name} Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return epoch_loss, accuracy

def main_train():
    model = TL_SDD_Model(num_classes=10).to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    print("Starting Phase 1: General classes only")
    train_loader_phase1 = get_data_loader(config["root_dir"], config["batch_size"], split='train', include_rare=False)
    for epoch in range(config["num_epochs_phase1"]):
        print(f"Epoch {epoch + 1}/{config['num_epochs_phase1']}")
        train_phase(model, train_loader_phase1, optimizer, criterion, "Phase 1")

    print("Starting Phase 2: General and rare classes")
    train_loader_phase2 = get_data_loader(config["root_dir"], config["batch_size"], split='train', include_rare=True)
    for epoch in range(config["num_epochs_phase2"]):
        print(f"Epoch {epoch + 1}/{config['num_epochs_phase2']}")
        train_phase(model, train_loader_phase2, optimizer, criterion, "Phase 2")

    torch.save(model.state_dict(), "tl_sdd_model.pth")
    print("Training complete and model saved as 'tl_sdd_model.pth'")

if __name__ == "__main__":
    main_train()
