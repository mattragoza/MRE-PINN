import numpy as np
import torch
import deepxde

from ..utils import minibatch
from ..pde import laplacian


class PINNData(deepxde.data.Data):

    def __init__(self, x, a, u, mu, pde, batch_size=None):

        self.x = x     # spatial coordinates
        self.a = a     # anatomic image
        self.u = u     # wave image
        self.mu = mu   # elastogram
        self.pde = pde # physical constraint

        self.batch_sampler = deepxde.data.BatchSampler(len(x), shuffle=True)
        self.batch_size = batch_size

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        (x, a), u_true = inputs, targets
        u_pred, mu_pred = outputs
        data_loss = loss_fn(u_true, u_pred)
        pde_res = self.pde(x, u_pred, mu_pred)
        pde_loss = loss_fn(0, pde_res)
        return [data_loss, pde_loss]

    def train_next_batch(self, batch_size=None):
        '''
        Args:
            batch_size
        Returns:
            inputs: A tuple of input arrays.
            targets: The target output array.
        '''
        batch_size = batch_size or self.batch_size
        if False: # patch sampling
            center_ind = self.batch_sampler.get_next(1)[0]
            batch_inds = np.argsort(self.dist[center_ind])[:batch_size]
        else:
            batch_inds = self.batch_sampler.get_next(batch_size)
        inputs = (self.x[batch_inds], self.a[batch_inds])
        targets = self.u[batch_inds]
        return inputs, targets

    def test(self):
        inputs = (self.x, self.a)
        targets = self.u
        return inputs, targets


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
        data = PINNData(x, a, u, mu, pde, batch_size)

        # initialize the network weights
        net.init_weights(inputs=(x, a), outputs=(u, mu))

        super().__init__(data, net)

    @minibatch
    def predict(self, x, a):

        # compute model predictions
        x = torch.as_tensor(x, dtype=torch.float32)
        a = torch.as_tensor(a, dtype=torch.float32)
        x.requires_grad_(True)
        u_pred, mu_pred = self.net((x, a))

        # compute differential operators
        lu_pred = laplacian(u_pred, x, dim=1)
        f_trac, f_body = self.data.pde.traction_and_body_forces(x, u_pred, mu_pred)
        deepxde.gradients.clear()

        return u_pred, lu_pred, mu_pred, f_trac, f_body
