import numpy as np
import torch
import deepxde

from .utils import minibatch
from . import pde


class MREPINNModel(deepxde.Model):

    def __init__(self, net, pde, geom, bc, batch_size=None, **kwargs):
        bc.set_batch_size(batch_size)
        data = deepxde.data.PDE(geom, pde, bc, **kwargs)
        super().__init__(data, net)

    @minibatch
    def predict(self, x):

        # compute model predictions
        x = torch.as_tensor(x)
        x.requires_grad_(True)
        outputs = self.net(x)
        u_pred, mu_pred = outputs[:,:-1], outputs[:,-1:]

        # compute differential operators
        lu_pred = pde.laplacian(u_pred, x, dim=1)
        f_trac, f_body = self.data.pde.traction_and_body_forces(x, outputs)
        deepxde.gradients.clear()

        return u_pred, lu_pred, mu_pred, f_trac, f_body


class PeriodicCallback(deepxde.callbacks.Callback):

    def __init__(self, period):
        super().__init__()
        self.period = period

    def on_batch_begin(self):
        if self.model.train_state.step % self.period != 0:
            return
        self.on_period_begin()

    def on_batch_end(self):
        if (self.model.train_state.step + 1) % self.period != 0:
            return
        self.on_period_end()

    def on_period_begin(self):
        pass

    def on_period_end(self):
        pass


class PDEResampler(PeriodicCallback):
    '''
    Resample PDE and BC training points each period.
    '''
    def on_period_end(self):
        self.model.data.train_x_all = None
        self.model.data.train_x_bc = None
        self.model.data.resample_train_points()


class SummaryDisplay(deepxde.display.TrainingDisplay):
    '''
    Training display that only prints a summary at
    the end of training instead of after every test.
    '''
    def print_one(self, *args, **kwargs):
        return


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
