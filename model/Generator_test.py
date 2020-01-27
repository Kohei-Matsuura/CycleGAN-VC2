import torch
x = torch.rand(30, 104, 40)

import Generator as G
m = G.Generator()

y = m(x)

import utils.tools as tools

print(y.shape)
tools.tensor_to_img(y[0].squeeze(0), 'gen_test.png')
