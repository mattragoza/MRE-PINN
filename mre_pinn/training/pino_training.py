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
        (a, x, y), (u_true,) = inputs, targets
        u_pred = outputs
        data_loss = loss_fn(u_true, u_pred)
        print(data_loss.shape)
        #pde_res = self.pde(x, u_pred, mu_pred)
        pde_loss = 0 #loss_fn(0, pde_res)
        return [data_loss, data_loss]

    def get_tensors(self, idx):
        '''
        Args:
            idx: Patient index in cohort.
        Returns:
            input: Tuple of input tensors.
            target: Tuple of target tensors.
        '''
        patient = self.cohort[idx]

        a_sequences = [
            't1_pre_in', 't1_pre_out', 't1_pre_water', 't1_pre_fat'
        ]
        a_arrays = [patient.arrays[seq].values for seq in a_sequences]
        a = np.stack(a_arrays, axis=-1)
        a = torch.tensor(a, device='cuda', dtype=torch.float32)

        x = patient.arrays['t1_pre_in'].field.points(reshape=False)
        x = torch.tensor(x, device='cuda', dtype=torch.float32)

        u = patient.arrays['wave'].values[...,None]
        u = torch.tensor(u, device='cuda', dtype=torch.float32)

        y = patient.arrays['wave'].field.points(reshape=False)
        y = torch.tensor(y, device='cuda', dtype=torch.float32)
        
        mu = patient.arrays['mre'].values[...,None]
        mu = torch.tensor(mu, device='cuda', dtype=torch.float32)

        z = patient.arrays['mre'].field.points(reshape=False)
        z = torch.tensor(z, device='cuda', dtype=torch.float32)

        return (a, x, y), u

    def train_next_batch(self, batch_size=None):
        '''
        Args:
            batch_size: Number of patients in batch.
        Returns:
            inputs: Tuple of input tensors.
            targets: Tuple of target tensors.
        '''
        batch_size = batch_size or self.batch_size
        batch_inds = self.batch_sampler.get_next(batch_size)

        inputs, targets = [], []
        for idx in batch_inds:
            input, target = self.get_tensors(idx)
            inputs.append(input)
            targets.append(target)

        inputs = tuple(torch.stack(x) for x in zip(*inputs))
        targets = torch.stack(targets)
        return inputs, targets

    def test(self):
        return self.train_next_batch()


class PINOModel(deepxde.Model):
    
    def __init__(self, cohort, net, pde, batch_size=None):

        # initialize the training data
        data = PINOData(cohort, pde, batch_size)

        # initialize the network weights
        #TODO net.init_weights()

        super().__init__(data, net)

    def predict(self, x, a):
        raise NotImplementedError('TODO')
