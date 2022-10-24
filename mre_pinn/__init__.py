import os
os.environ['DDEBACKEND'] = 'pytorch'
import deepxde

from . import (
	data,
	model,
	training,
	testing,
	fields,
	pde,
	discrete,
	fem,
	visual,
	utils
)
