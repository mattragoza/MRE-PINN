import sys, os
import numpy as np
import torch

os.environ['DDEBACKEND'] = 'pytorch'
import deepxde

import mre_pinn
from mre_pinn.utils import main


@main
def train(

    # data settings
    data_root='data/BIOQIC',
    data_name='fem_box',
    frequency=80,
    xyz_slice='2D',
    noise_ratio=0.0,

    # pde settings
    pde_name='hetero',

    # model settings
    omega0=16,
    n_layers=5,
    n_hidden=128,
    activ_fn='t',
    polar=False,
    conditional=False,

    # training settings
    optimizer='adam',
    learning_rate=1e-4,
    pde_loss_wt=1e-8,
    data_loss_wt=1e0,
    batch_size=128,
    n_iters=100000,

    # testing settings
    test_every=1000,
    save_every=10000,
    save_prefix=None
):
    # load the training data
    data = mre_pinn.data.load_bioqic_dataset(
        data_root=data_root,
        data_name=data_name,
        frequency=frequency,
        xyz_slice=xyz_slice,
        noise_ratio=noise_ratio
    )

    # define model architecture
    pinn = mre_pinn.model.ParallelPINN(
        n_inputs=[data.field.n_spatial_dims + 1, data.field.n_spatial_dims],
        n_outputs=[data.field.n_spatial_dims, 1],
        omega0=omega0,
        n_layers=n_layers,
        n_hidden=n_hidden,
        activ_fn=activ_fn,
        dense=True,
        polar=polar,
        conditional=conditional,
        dtype=torch.float32
    )
    print(pinn)

    # define PDE that we want to solve
    pde = mre_pinn.pde.WaveEquation.from_name(pde_name, detach=True)

    # compile model and configure training settings
    model = mre_pinn.training.PINNModel(data, pinn, pde, batch_size)
    model.compile(
        optimizer=optimizer,
        lr=learning_rate,
        loss_weights=[pde_loss_wt, data_loss_wt],
        loss=mre_pinn.training.losses.standardized_msae_loss_fn(data.u.values)
    )
    test_eval = mre_pinn.testing.TestEvaluator(
        test_every=test_every,
        save_every=save_every,
        save_prefix=save_prefix
    )

    # train the model
    model.train(n_iters, display_every=10, callbacks=[test_eval])

    # final test evaluation
    print('Final test evaluation')
    test_eval.test(data)
    print(test_eval.metrics)
