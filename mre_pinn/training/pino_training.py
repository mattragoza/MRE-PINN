import numpy as np
import torch
import deepxde


class PINOData(deepxde.data.Data):

	def __init__(self, cohort, pde, batch_size=None):

		self.cohort = cohort
		self.pde = pde

		self.batch_sampler = deepxde.data.BatchSampler(len(cohort), shuffle=True)
		self.batch_size = batch_size

	def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
		return [data_loss, pde_loss]

	def train_next_batch(self, batch_size=None):
		'''
		Args:
			batch_size
		Returns:
			inputs: (batch_size, patch_size, n_input) input arrays
			targets: (batch_size, patch_size, n_output) target arrays
		'''
		batch_size = batch_size or self.batch_size
		batch_inds = self.batch_sampler.get_next(batch_size)

		for idx in batch_inds:
			patient = self.cohort[idx]
			anat_sequences = ['t1_pre_water', 't1_pre_out', 't1_pre_fat', 't2']
			a = np.stack([patient.arrays[seq].values for seq in anat_sequences])
			a = torch.tensor(a, device='cuda', dtype=torch.float32)
			u = patient.arrays['wave']
			u = torch.tensor(u, device='cuda', dtype=torch.float32)
			mu = patient.arrays['mre']
			mu = torch.tensor(mu, device='cuda', dtype=torch.float32)

			x = patient.arrays['mask'].field.points()
			mask = patient.arrays['mask'].field.values()

		return inputs, targets

	def test(self):
		return inputs, targets


class PINOModel(deepxde.Model):
	
	def __init__(self, cohort, net, pde, batch_size=None):

		# initialize the training data
		data = PINOData(cohort, pde, batch_size)

		# initialize the network weights
		net.init_weights()

		super().__init__(data, net)

	def predict(self, x, a):
		raise NotImplementedError('TODO')
