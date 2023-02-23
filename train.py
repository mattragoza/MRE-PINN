import sys, os
import numpy as np
import torch

os.environ['DDEBACKEND'] = 'pytorch'
import deepxde

import mre_pinn
from mre_pinn.utils import main
from mre_pinn.training.losses import msae_loss


@main
def train(

    # data settings
    xarray_dir='data/BIOQIC/fem_box',
    example_id='60',
    frequency='auto',
    noise_ratio=0.0,
    anatomical=False,

    # pde settings
    pde_name='hetero',

    # model settings
    omega=30,
    n_layers=5,
    n_hidden=128,
    activ_fn='s',
    polar_input=False,

    # training settings
    optimizer='adam',
    learning_rate=1e-4,
    u_loss_weight=1.0,
    mu_loss_weight=0.0,
    a_loss_weight=0.0,
    pde_loss_weight=1e-16,
    pde_warmup_iters=10000,
    pde_init_weight=1e-18,
    pde_step_iters=5000,
    pde_step_factor=10,
    n_points=1024,
    n_iters=100000,

    # testing settings
    test_every=1000,
    save_every=10000,
    save_prefix=None
):
    # load the training data
    example = mre_pinn.data.MREExample.load_xarrays(
        xarray_dir=xarray_dir,
        example_id=example_id,
        anat=anatomical
    )
    if frequency == 'auto': # infer from data
        frequency = float(example.wave.frequency.item())
    else:
        frequency = float(frequency)

    if noise_ratio > 0:
        example.add_gaussian_noise(noise_ratio)

    mre_pinn.baseline.eval_ahi_baseline(example, frequency=frequency)
    mre_pinn.baseline.eval_fem_baseline(
        example,
        frequency=frequency,
        hetero=(pde_name == 'hetero'),
        hetero2=(pde_name == 'hetero2')
    )

    # define PDE that we want to solve
    pde = mre_pinn.pde.WaveEquation.from_name(
        pde_name, omega=frequency, detach=True
    )

    # define the model architecture
    pinn = mre_pinn.model.MREPINN(
        example,
        omega=omega,
        n_layers=n_layers,
        n_hidden=n_hidden,
        polar_input=polar_input
    )
    print(pinn)

    # compile model and configure training settings
    model = mre_pinn.training.MREPINNModel(
        example, pinn, pde,
        loss_weights=[u_loss_weight, mu_loss_weight, a_loss_weight, pde_loss_weight],
        pde_warmup_iters=pde_warmup_iters,
        pde_step_iters=pde_step_iters,
        pde_step_factor=pde_step_factor,
        pde_init_weight=pde_init_weight,
        n_points=n_points
    )
    model.compile(optimizer='adam', lr=learning_rate, loss=msae_loss)
    model.benchmark(100)

    test_eval = mre_pinn.testing.TestEvaluator(
        test_every=test_every,
        save_every=save_every,
        save_prefix=save_prefix
    )
    # train the model
    model.train(n_iters, display_every=10, callbacks=[test_eval])

    # final test evaluation
    print('Final test evaluation')
    test_eval.test()
    print(test_eval.metrics)

    print('Done')
