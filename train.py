import sys, os, fire
import numpy as np
import torch

os.environ['DDEBACKEND'] = 'pytorch'
import deepxde

import mre_pinn


def train(

    # data settings
    data_root='data/BIOQIC',
    data_name='fem_box',
    frequency=80,
    xyz_slice='2D',
    noise_ratio=0,

    # pde settings
    pde_name='hetero',

    # model settings
    omega0=16,
    n_layers=5,
    n_hidden=128,
    activ_fn='t',

    # training settings
    learning_rate=1e-4,
    pde_loss_wt=1e-8,
    data_loss_wt=1,
    optimizer='adam',
    batch_size=80,
    pde_distrib='pseudo',
    n_domain=48,
    n_iters=100000,

    # testing settings
    test_every=1000,
    save_every=10000,
    save_prefix=None
):
    data, test_data = mre_pinn.data.load_bioqic_dataset(
        data_root=data_root,
        data_name=data_name,
        frequency=frequency,
        xyz_slice=xyz_slice,
        noise_ratio=noise_ratio
    )

    # convert to vector/scalar fields and coordinates
    x  = data.u.field.points().astype(np.float32)
    u  = data.u.field.values().astype(np.complex64)
    mu = data.mu.field.values().astype(np.complex64)

    # initialize the PDE, geometry, and boundary conditions
    pde = mre_pinn.pde.WaveEquation.from_name(pde_name, detach=True)
    geom = deepxde.geometry.Hypercube(x.min(axis=0), x.max(axis=0) + 1e-5)
    bc = mre_pinn.fields.VectorFieldBC(points=x, values=u)

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
    model = mre_pinn.training.MREPINNModel(
        net, pde, geom, bc,
        batch_size=batch_size,
        num_domain=n_domain,
        num_boundary=0,
        train_distribution=pde_distrib,
        anchors=None
    )
    model.compile(
        optimizer=optimizer,
        lr=learning_rate,
        loss_weights=[pde_loss_wt, data_loss_wt],
        loss=mre_pinn.training.standardized_msae_loss_fn(u)
    )
    test_eval = mre_pinn.training.TestEvaluation(
        data=test_data,
        batch_size=batch_size,
        test_every=test_every,
        save_every=save_every,
        save_prefix=save_prefix
    )
    sampler = mre_pinn.training.PDEResampler(period=1)

    # train the model
    model.train(n_iters, display_every=10, callbacks=[test_eval, sampler])

    # final test evaluation
    test_eval.test_evaluate(data)


if __name__ == '__main__':
    fire.Fire(train)
