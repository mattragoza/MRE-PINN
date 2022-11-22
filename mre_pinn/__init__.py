import os
os.environ['DDEBACKEND'] = 'pytorch'
import deepxde

from . import (
	data,
	model,
	training,
	testing,
	baseline,
	fields,
	pde,
	visual,
	utils
)
