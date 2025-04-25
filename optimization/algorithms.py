import torch.optim as optim

def get_optimizer(parameters, optimizer_name='SGD', lr=0.01):
    if optimizer_name == 'SGD':
        return optim.SGD(parameters, lr=lr)
    elif optimizer_name == 'Adam':
        return optim.Adam(parameters, lr=lr)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported")
