import torch
from multi_head_test import MultiHeadAttention
from torch import nn



criterion = nn.CrossEntropyLoss()
x =torch.ones(1,5,10)
attention = MultiHeadAttention(d_model=10, n_head=2)
out,gradient_wo,_,_,_ = attention(q=x, k=x, v=x)


loss = out.mean()
loss.backward()


wo_gradient = attention.w_concat.weight.grad
print(wo_gradient)
print()
print('w^o error is {}'.format(gradient_wo-wo_gradient))



