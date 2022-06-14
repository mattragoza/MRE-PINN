import sys, os
import numpy as np
import torch

os.environ['DDEBACKEND'] = 'pytorch'
import deepxde

import mre_pinn


def train(
	wave_file='data/wave_sim/steady_state_wave.npy',
	elast_file='data/wave_sim/elasticity.npy',
	n_layers=5,
	n_hidden=128,
	activ_fn='sin',
	lr=1e-3,
	pde_loss_wt=1e-5,
	data_loss_wt=1,
	optimizer='adam',
	n_domain=200,
	n_test=200,
	n_iters=20000,
):
	print(f'Loading wave image {wave_file}')
	wave_image = mre_pinn.data.ImagePointSet(wave_file)

	print(f'Loading elastogram {elast_file}')
	elast_image = mre_pinn.data.ImagePointSet(elast_file)
	
	print('Initializing PDE and geometry')
	pde = mre_pinn.pde.HelmholtzPDE(rho=1, omega=10*(2*np.pi))
	geometry = deepxde.geometry.Rectangle([0, 0], [1, 1])

	# data combines geometry, PDE residual, and boundary conditions
	data = deepxde.data.PDE(
		geometry=geometry,
		pde=pde,
		bcs=[wave_image],
		num_domain=n_domain,
		anchors=wave_image.points,
		num_test=n_test
	)

	print('Initializing model')
	net = mre_pinn.model.Parallel([
		mre_pinn.model.ComplexFFN(
			n_input=2,
			n_layers=n_layers,
			n_hidden=n_hidden,
			n_output=n_output,
			activ_fn={'sin': torch.sin}[activ_fn]
		) for n_output in [2, 1] # u and mu
	])
	print(net)

	model = deepxde.Model(data, net)
	model.compile(
		optimizer=optimizer, lr=lr, loss_weights=[pde_loss_wt, data_loss_wt]
	)

	print('Starting training loop')
	loss_history, train_state = model.train(epochs=n_iters)

	# TODO evaluate results
	print('Done')


if __name__ == '__main__':
	train()
