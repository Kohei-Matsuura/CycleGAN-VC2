import torch

x = torch.rand(30, 1, 128, 40)

import Discriminator as D
m = D.Discriminator()

y = m(x)

#print(y0)
print(y.shape)
import utils.tools as tools
tools.tensor_to_img(y[0].squeeze(0), 'disc_test.png')
