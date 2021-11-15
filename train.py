import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, dataloader
from dataset import ModelNet, ShapeNetPart, point_Random_Jitter, point_Random_Rotate, point_Scale
from model import feature_transform_regularizer, PointNet_Classification
from utils.util import cal_loss
from torchvision import transforms


def train_cls(args=None):
    seed = args.manual_seed
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        args.use_gpus = True

    device = torch.device("cuda" if args.use_gpus else "cpu")

    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory USage:")
        print(f"Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB")
        print(f"Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB")
        torch.cuda.manual_seed_all(seed)

    transform = transforms.Compose([point_Random_Rotate(), point_Random_Jitter(), point_Scale(),])

    train_set = ModelNet(train=True, num_points=args.num_points, transform=transform)
    test_set = ModelNet(train=False, num_points=args.num_points, transform=transform)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,)
    num_classes = len(train_set.classes)
    args.num_classes = num_classes
    pointnet_model = PointNet_Classification(args=args, num_classes=num_classes)
    optimizer = optim.Adam(pointnet_model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.5)
    pointnet_model.to(device)
    print(pointnet_model)
    num_batch=len(train_set)/args.batch_size

    for epoch in range(args.num_epochs):
        scheduler.step()
        for idx, data in enumerate(train_dataloader):
            points, target = data
            points = points.transpose(2, 1)
            target = target[:, 0]
            points, target = points.to(device), target.to(device)
            optimizer.zero_grad()
            pointnet_model.train()
            pred, trans, trans_feat = pointnet_model(points)
            loss = F.nll_loss(pred, target)
            if args.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct=pred_choice.eq(target.data).cpu().sum()
            print(f"[{epoch}: {idx}/{num_batch}] - Loss: {loss.item()} , Accuracy: {correct.item()/float(args.batch_size)}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual_seed", type=str, default=42, help="Inital Seed")
    parser.add_argument(
        "--class_choice",
        type=str,
        default=None,
        choices=[
            "airplane",
            "bag",
            "cap",
            "car",
            "chair",
            "earphone",
            "guitar",
            "knife",
            "lamp",
            "laptop",
            "motor",
            "mug",
            "pistol",
            "rocket",
            "skateboard",
            "table",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
    parser.add_argument("--model", type=str, default="pointnet", help="Model architecture to be used")
    parser.add_argument("--num_points", type=int, default=1024, help="Number of Points")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of Epoches")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of Workers")
    parser.add_argument("--use_gpus", type=bool, default=False, help="Use GPUS")
    parser.add_argument("--feature_transform", type=bool, default=True, help="Use Feature Transforms")

    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning Rate (default:0.1 if use_sgd  else 0.001",
    )

    args = parser.parse_args()
    train_cls(args)


if __name__ == "__main__":
    main()
