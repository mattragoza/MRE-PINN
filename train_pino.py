import sys, os
import numpy as np
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
os.environ['DDEBACKEND'] = 'pytorch'
import deepxde

import mre_pinn
from mre_pinn.utils import main
from mre_pinn.training.losses import msae_loss


@main
def train(

    # data settings
    xarray_dir='data/BIOQIC/fem_box',
    frequency='auto',

    # pde settings
    pde_name='hetero',

    # model settings
    n_channels_block=16,
    n_conv_per_block=2,
    n_conv_blocks=5,
    activ_fn='g',
    n_latent=128,
    width_factor=2,
    n_pinn_layers=5,
    n_pinn_hidden=128,
    polar_input=True,
    conditional=True,
    parallel=False,
    omega=60,

    # training settings
    optimizer='adam',
    learning_rate=1e-5,
    u_loss_weight=1,
    mu_loss_weight=0,
    pde_loss_weight=1e-16,
    pde_warmup_iters=20000,
    pde_init_weight=1e-18,
    pde_step_iters=10000,
    pde_step_factor=10,
    n_points=4096,
    batch_size=8,
    n_iters=100000,

    # testing settings
    test_every=1000,
    save_every=10000,
    save_prefix=None
):
    # load the training data
    dataset = mre_pinn.data.MREDataset.load_xarrays(xarray_dir=xarray_dir, anat=True)
    if frequency == 'auto': # infer from data
        frequency = None
    else:
        frequency = float(frequency)

    for i in range(len(dataset)):
        mre_pinn.baseline.eval_direct_baseline(dataset[i], frequency=frequency)

    # train-test split
    dataset.shuffle(0)
    train_set, test_set = dataset[:-15], dataset[-15:]
    print(train_set.example_ids)
    print(test_set.example_ids)

    # define PDE that we want to solve
    pde = mre_pinn.pde.WaveEquation.from_name(pde_name, omega=frequency, detach=True)

    # define the model architecture
    pino = mre_pinn.model.MREPINO(
        train_set,
        n_channels_block=(n_channels_block, n_channels_block),
        n_conv_per_block=(n_conv_per_block, n_conv_per_block),
        n_conv_blocks=(n_conv_blocks, n_conv_blocks),
        activ_fn=activ_fn,
        n_latent=n_latent,
        width_factor=width_factor,
        n_pinn_layers=n_pinn_layers,
        n_pinn_hidden=n_pinn_hidden,
        polar_input=polar_input,
        conditional=conditional,
        parallel=parallel,
        omega=omega
    )
    print(pino)

    # compile model and configure training settings
    model = mre_pinn.training.MREPINOModel(
        train_set, test_set, pino, pde,
        loss_weights=[u_loss_weight, mu_loss_weight, pde_loss_weight],
        pde_warmup_iters=pde_warmup_iters,
        pde_step_iters=pde_step_iters,
        pde_step_factor=pde_step_factor,
        pde_init_weight=pde_init_weight,
        n_points=n_points,
        batch_size=batch_size
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
