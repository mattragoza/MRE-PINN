import sys, os, fire
import numpy as np

import mre_pinn


def solve(

    # data settings
    data_root='data/BIOQIC',
    data_name='fem_box',
    frequency=80,
    xyz_slice='2D',
    noise_ratio=0,

    # pde settings
    pde_name='helmholtz',

    # FEM settings
    u_elem_type='CG-1',
    mu_elem_type='CG-1',
    align_nodes=False,
    savgol_filter=False,

    # other settings
    save_prefix=None
):
    data, _ = mre_pinn.data.load_bioqic_dataset(
        data_root=data_root,
        data_name=data_name,
        frequency=frequency,
        xyz_slice=xyz_slice,
        noise_ratio=noise_ratio
    )

    fem = mre_pinn.fem.MultiFEM(
        data,
        u_elem_type=u_elem_type,
        mu_elem_type=mu_elem_type,
        align_nodes=align_nodes,
        savgol_filter=savgol_filter
    )
    if pde_name == 'helmholtz':
        fem.solve(homogeneous=True)
    elif pde_name == 'hetero':
        fem.solve(homogeneous=False)
    else:
        raise ValueError(f'unrecognized pde_name: {pde_name}')

    test_eval = mre_pinn.testing.TestEvaluator(data, fem, save_prefix=save_prefix)
    test_eval.test(data, save_model=False)
    print(test_eval.metrics)


if __name__ == '__main__':
    fire.Fire(solve)
