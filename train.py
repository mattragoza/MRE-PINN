import sys, os, fire
import numpy as np
import torch

os.environ['DDEBACKEND'] = 'pytorch'
import deepxde

import mre_pinn


def train(

    # data settings
    data_root='data/BIOQIC',
    downsample=False,
    frequency=None,
    x_slice=None,
    y_slice=None,
    z_slice=None,

    # pde settings
    pde_name='helmholtz',

    # model settings
    omega0=32,
    n_layers=5,
    n_hidden=128,
    activ_fn='s',

    # training settings
    learning_rate=1e-4,
    pde_loss_wt=1e-7,
    data_loss_wt=1,
    optimizer='adam',
    batch_size=200,
    n_domain=200,
    n_iters=20000
):
    # load the FEM box data set
    data = mre_pinn.data.load_bioqic_fem_box_data(data_root)

    # select data subset
    data, ndim = mre_pinn.data.select_data_subset(
        data, downsample, frequency, x_slice, y_slice, z_slice
    )
    print(data)

    # direct Helmholtz inversion via discrete laplacian
    data['Lu'] = mre_pinn.discrete.laplacian(data['u'])
    data['Mu'] = mre_pinn.discrete.helmholtz_inversion(data['u'], data['Lu'])

    # test on 4x downsampled data
    test_data = data.coarsen(**{d: 4 for d in data.field.spatial_dims}).mean()

    # convert to vector/scalar fields and coordinates
    x  = data.u.field.points().astype(np.float32)
    u  = data.u.field.values().astype(np.complex64)
    mu = data.mu.field.values().astype(np.complex64)

    print('x ', type(x), x.shape, x.dtype)
    print('u ', type(u), u.shape, u.dtype)
    print('mu', type(mu), mu.shape, mu.dtype)

    # initialize the PDE, boundary conditions, and geometry
    pde = mre_pinn.pde.WaveEquation.from_name(pde_name, detach=True)
    bc = mre_pinn.data.PointSetBC(x, u, batch_size=batch_size)
    geom = deepxde.geometry.Hypercube(x.min(axis=0), x.max(axis=0))

    # define model architecture
    net = mre_pinn.model.MREPINN(
        input=x,
        outputs=[u, mu],
        omega0=omega0,
        n_layers=n_layers,
        n_hidden=n_hidden,
        activ_fn=activ_fn,
        parallel=True,
        dense=True,
        dtype=torch.float32
    )
    print(net)

    # compile model and configure training settings
    model = mre_pinn.training.MREPINNModel(net, pde, geom, bc, num_domain=batch_size)
    model.compile(
        optimizer=optimizer,
        lr=learning_rate,
        loss_weights=[pde_loss_wt, data_loss_wt],
        loss=mre_pinn.training.normalized_l2_loss_fn(u)
    )
    #deepxde.display.training_display = mre_pinn.visual.TrainingPlot(
    #    losses=['pde_loss', 'data_loss'], metrics=[]
    #)
    callbacks = [
        mre_pinn.training.TestEvaluation(100, test_data, batch_size, col='frequency'),
        mre_pinn.training.PDEResampler(period=1),
    ]

    # train the model
    model.train(n_iters, display_every=10, callbacks=callbacks)


if __name__ == '__main__':
    fire.Fire(train)
