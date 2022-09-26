import os
os.environ['DDEBACKEND'] = 'pytorch'
import deepxde

from . import (
	data,
	fields,
	pinn,
	pino,
	pde,
	training,
	testing,
	discrete,
	visual,
	utils,
	fem
)
