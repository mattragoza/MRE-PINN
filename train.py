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
    data_root,
    example_id,
    frequency,

    # pde settings
    pde_name='hetero',
    pde_warmup_iters=10000,
    pde_init_weight=1e-19,
    pde_step_iters=5000,
    pde_step_factor=10,

    # model settings
    omega=16,
    n_layers=4,
    n_hidden=128,
    activ_fn='s',
    conditional=False,

    # training settings
    optimizer='adam',
    learning_rate=1e-4,
    u_loss_wt=1,
    mu_loss_wt=0,
    pde_loss_wt=1e-8,
    n_points=1024,
    n_iters=100000,

    # testing settings
    test_every=1000,
    save_every=10000,
    save_prefix=None
):
    # load the training data
    example = mre_pinn.data.MREExample.load_xarrays(
        data_root=data_root,
        example_id=example_id
    )
    example.eval_baseline(frequency=frequency, polar=True)

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
        polar_input=polar_input,
        conditional=conditional
    )
    print(pinn)

    # compile model and configure training settings
    model = mre_pinn.training.MREPINNModel(
        example, pinn, pde,
        loss_weights=[u_loss_wt, mu_loss_wt, pde_loss_wt],
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
