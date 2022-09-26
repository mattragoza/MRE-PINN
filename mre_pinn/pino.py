import torch
from torch import nn


class PINO(torch.nn.ModuleList):
	'''
	Physics-informed neural operator.
	'''

	def __init__(
		self,
		n_input,
		n_blocks,
		n_output
	):
		super().__init__()

		self.blocks = []
		for i in range(n_blocks):
			block = PINOBlock(
				d_model,
				n_heads,
				d_ff,
				activ_fn,
				dropout
			)
			self.blocks.append(block)
			self.add_module(f'block{i}', block)

	def forward(self, a, x, y):
		'''
		Args:
			a: (N, D_a) input codomain samples.
			x: (N, D_x) input domain samples.
			y: (M, D_y) output domain samples.
		Returns:
			u: (M, D_u) output codomain samples.
		'''
		h = y
		for block in self.blocks:
			h = block(a, x, h)
		return h


class PINOBlock(torch.nn.ModuleList):

	def __init__(self, d_model, n_heads, d_ff, activ_fn, dropout=0.0):
		self.attention = nn.MultiHeadAttention(
			embed_dim=d_model,
			num_heads=n_heads,
			dropout=dropout
		)
		self.feedforward = FeedForward(
			n_input=d_model,
			n_hidden=d_ff,
			n_output=d_model,
			activ_fn=activ_fn,
			dropout=dropout
		)
		self.layer_norm1 = nn.LayerNorm(d_model)
		self.layer_norm2 = nn.LayerNorm(d_model)

	def forward(self, a, x, h):
		'''
		Args:
			a: (N, D_a) input codomain samples.
			x: (N, D_x) input domain samples.
			h: (N, D_h) hidden representation.
		Returns:
			h': (N, D_h) new hidden representation.
		'''
		h = self.layer_norm1(self.attention(query=h, key=x, value=a) + h)
		return self.layer_norm2(self.feedforward(input=h) + h)


class FeedForward(torch.nn.Sequential):
	
	def __init__(self, n_input, n_hidden, n_output, activ_fn, dropout=0.0):
		self.linear1 = nn.Linear(n_input, n_hidden)
		self.dropout1 = nn.Dropout(dropout)
		self.activ_fn = activ_fn
		self.linear2 = nn.Linear(n_hidden, n_output)
		self.dropout2 = nn.Dropout(dropout)

	def forward(self, input):
		hidden = self.dropout1(self.activ_fn(self.linear1(input)))
		return self.dropout2(self.linear2(hidden))
