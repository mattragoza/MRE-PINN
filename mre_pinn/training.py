import numpy as np
from scipy.spatial import distance
import torch
import deepxde

from .utils import minibatch
from . import pde, fields


class MREData(deepxde.data.Data):

    def __init__(self, x, a, u, mu, pde, batch_size=None):

        self.x = x     # spatial coordinates
        self.a = a     # anatomic image
        self.u = u     # wave image
        self.mu = mu   # elastogram
        self.pde = pde # physical constraint

        # compute pairwise distance between points
        self.dist = distance.squareform(distance.pdist(x))

        self.batch_sampler = deepxde.data.BatchSampler(len(x), shuffle=True)
        self.batch_size = batch_size

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        (x, a), u_true = inputs, targets
        u_pred, mu_pred = outputs[:,:-1], outputs[:,-1:]
        data_loss = loss_fn(u_true, u_pred)
        pde_res = self.pde(x, outputs)
        pde_loss = loss_fn(0, pde_res)
        return [data_loss, pde_loss]

    def train_next_batch(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        if False: # patch sampling
            center_ind = self.batch_sampler.get_next(1)[0]
            batch_inds = np.argsort(self.dist[center_ind])[:batch_size]
        else:
            batch_inds = self.batch_sampler.get_next(batch_size)
        return (self.x[batch_inds], self.a[batch_inds]), self.u[batch_inds]

    def test(self):
        return (self.x, self.a), self.u


class PINNModel(deepxde.Model):

    def __init__(self, data, net, pde, batch_size=None):

        # convert to vector/scalar fields and coordinates
        #   while masking out the background region
        region = data.spatial_region.field.values()[:,0]
        x = data.field.points().astype(np.float32)[region >= 0]
        a = data.a.field.values().astype(np.float32)[region >= 0]
        u = data.u.field.values().astype(np.complex64)[region >= 0]
        mu = data.mu.field.values().astype(np.complex64)[region >= 0] 

        # initialize the training data
        data = MREData(x, a, u, mu, pde, batch_size)

        # initialize the network weights
        net.init_weights(inputs=(x, a), outputs=(u, mu))

        super().__init__(data, net)

    @minibatch
    def predict(self, x, a):

        # compute model predictions
        x = torch.as_tensor(x, dtype=torch.float32)
        a = torch.as_tensor(a, dtype=torch.float32)
        x.requires_grad_(True)
        outputs = self.net((x, a))
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
