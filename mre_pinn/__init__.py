import os
os.environ['DDEBACKEND'] = 'pytorch'
import deepxde

from . import (
	data, model, training, fields, pde, discrete, visual, utils, fem
)
