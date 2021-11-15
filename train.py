import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, dataloader
from dataset import ModelNet, ShapeNetPart, point_Random_Jitter, point_Random_Rotate, point_Scale
from model import PointNet
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
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,)


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
    parser.add_argument("--num_points", type=int, default=1024, help="Number of Points")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of Epoches")
    parser.add_argument("--model", type=str, default="pointnet", help="Model architecture to be used")
    parser.add_argument("--use_gpus", type=bool, default=False, help="Use GPUS")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of Workers")

    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning Rate (default:0.1 if use_sgd  else 0.001",
    )
    parser.add_argument("--scheduler", type=str, default="step", help="Learning Rate Scheduler")

    parser.add_argument("--use_sgd", type=str, default=False, help="Use SGD optimizer")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD Momentum")
    parser.add_argument("--weight_deacy", type=float, default=1e-4, help="Weight Decay")

    args = parser.parse_args()
    train_cls(args)


if __name__ == "__main__":
    main()
