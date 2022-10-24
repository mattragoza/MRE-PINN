import numpy as np
import torch


def normalized_l2_loss_fn(y):
    norm = np.linalg.norm(y, axis=1).mean()
    def loss_fn(y_true, y_pred):
        return torch.mean(
            torch.norm(y_true - y_pred, dim=1) / norm
        )
    return loss_fn


def standardized_msae_loss_fn(y):
    variance = torch.var(torch.as_tensor(y))
    def loss_fn(y_true, y_pred):
        return torch.mean(
            torch.abs(y_true - y_pred)**2 / variance
        )
    return loss_fn
