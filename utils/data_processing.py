import torch
from torchvision import datasets, transforms

def load_fashion_mnist(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.FashionMNIST('.', download=True, train=True, transform=transform)
    test_set = datasets.FashionMNIST('.', download=True, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    class_names = train_set.classes

    return train_loader, test_loader, class_names
