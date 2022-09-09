import os
os.environ['DDEBACKEND'] = 'pytorch'
import deepxde

from . import (
	data,
	fields,
	model,
	pde,
	training,
	testing,
	discrete,
	visual,
	utils,
	fem
)
