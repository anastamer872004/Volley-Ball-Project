import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


group_activities = {
    "r_spike": 0,
    "r_set": 1,
    "r-pass": 2,
    "r_winpoint": 3,
    "l-spike": 4,
    "l_set": 5,
    "l-pass": 6,
    "l_winpoint": 7,
}


splits = {
    "train": [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
    "val": [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
    "test": [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47],
}


class VolleyballDataset(Dataset):
    def __init__(self, videos_dir, annotations_dir, split, splits, transform=None):
        self.data = []
        self.transform = transform

        for video_id in splits[split]: # 1
            video_path = os.path.join(videos_dir, str(video_id)) # D:\Ai\DL\Projects\VolleyBall\VideosSplit\Videos\1
            annotation_path = os.path.join(annotations_dir, str(video_id)) # D:\Ai\DL\Projects\VolleyBall\VideosSplit\Annotations\1

            clip_annotation = os.path.join(annotation_path, "annotations.txt") # D:\Ai\DL\Projects\VolleyBall\VideosSplit\Annotations\1\annotations.txt

            with open(clip_annotation, 'r') as file:
                   
                clip_category_dct = {}

                for line in file:
                  items = line.strip().split(' ')[:2]
                  clip_dir = items[0].replace('.jpg', '')
                  clip_category_dct[clip_dir] = items[1]

            for clip_id in os.listdir(video_path): # 9530
                clip_folder = os.path.join(video_path, clip_id) # D:\Ai\DL\Projects\VolleyBall\VideosSplit\Videos\1\9530

                activity = clip_category_dct[clip_id] # r_winpoint

                middle_frame = f"{clip_id}.jpg" # 9530.jpg
                image_path = os.path.join(clip_folder, middle_frame) # D:\Ai\DL\Projects\VolleyBall\VideosSplit\Videos\1\9530.jpg

                self.data.append((image_path, activity)) 
                print(self.data)
            break
                

        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, activity = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        label = torch.tensor(group_activities[activity])

        if self.transform:
            image = self.transform(image)

        return image, label


class Baseline1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / len(loader), acc, f1


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / len(loader), acc, f1, all_labels, all_preds


def main():
    videos_dir = r"D:\Ai\DL\Projects\VolleyBall\VideosSplit\Videos"
    annotations_dir = r"D:\Ai\DL\Projects\VolleyBall\VideosSplit\Annotations"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = VolleyballDataset(videos_dir, annotations_dir, "train", splits, transform)
    # val_dataset = VolleyballDataset(videos_dir, annotations_dir, "val", splits, transform)
    # test_dataset = VolleyballDataset(videos_dir, annotations_dir, "test", splits, transform)

    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)
    # test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # print(f"Train samples: {len(train_dataset)}")
    # print(f"Val samples: {len(val_dataset)}")
    # print(f"Test samples: {len(test_dataset)}")

    # model = Baseline1(num_classes=len(group_activities)).to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    
    # best_val_acc = 0
    # for epoch in range(5):
    #     train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
    #     val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

    #     print(f"Epoch {epoch+1}/5")
    #     print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | F1: {train_f1:.2f}")
    #     print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.2f}")

    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         torch.save(model.state_dict(), "baseline1_best.pth")
    #         print(" Best model saved!")


    # model.load_state_dict(torch.load("baseline1_best.pth"))
    # test_loss, test_acc, test_f1, test_labels, test_preds = evaluate(model, test_loader, criterion, device)
    # print("\nFinal Test Results:")
    # print(f"  Test Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% | F1: {test_f1:.2f}")

   
    # cm = confusion_matrix(test_labels, test_preds)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(group_activities.keys()))
    # disp.plot(cmap="Blues", xticks_rotation=45)
    # plt.title("Confusion Matrix - Test Set")
    # plt.show()

if __name__ == "__main__":
    main()
