import torch
from torch import nn, optim
from torchvision.datasets import CIFAR10
from torchvision.models import resnet152
from torchvision.transforms import ToTensor

from src.core import find_best_batch_size

DEVICE = torch.device("cuda")


def main():
    print(f"Device: {DEVICE}")

    model = resnet152()

    optimizer = optim.AdamW(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    dataset = CIFAR10("./data", transform=ToTensor(), download=True)

    max_bs = find_best_batch_size(
        model,
        optimizer,
        loss_fn,
        dataset,
        device=DEVICE,
        verbose=True,
    )

    print(f"Best batch size: {max_bs}")


if __name__ == "__main__":
    main()
