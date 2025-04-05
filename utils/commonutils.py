import yaml
import torch

def load_yaml(config_path, encoding='utf-8'):
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.safe_load(f)

def smooothing_loss(y_pred):
    dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
    dy = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
    dx = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])

    dx = dx * dx
    dy = dy * dy
    dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    grad = d
    return d

def smooothing_loss2D(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
    dx = dx * dx
    dy = dy * dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d
    return d