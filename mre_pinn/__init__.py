import os
os.environ['DDEBACKEND'] = 'pytorch'
import deepxde

from . import (
	data, model, pde, training, discrete, visual, utils
)
