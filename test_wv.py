import torch
from multi_head_test import MultiHeadAttention
from torch import nn

def concat(tensor):
    """
    inverse function of self.split(tensor : torch.Tensor)

    :param tensor: [batch_size, head, length, d_tensor]
    :return: [batch_size, length, d_model]
    """
    batch_size, head, length, d_tensor = tensor.size()
    d_model = head * d_tensor

    tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
    return tensor
    
x =torch.ones(1,5,10)
attention = MultiHeadAttention(d_model=10, n_head=2)
out,gradient_wo,att1,_,_ = attention(q=x, k=x, v=x)
x=x.squeeze()
att1=concat(att1@x).squeeze()
I=torch.block_diag(torch.ones(10,5),torch.ones(10,5)) # I extract matrix to rxtract what we want
# print(attention.w_concat.weight)
w_vtheory = (att1.transpose(0,1)@(torch.ones(5,10)/50)@attention.w_concat.weight)*I # nn.linear: y =x@w^T+bias,so we don't need tranpose W^O

i,j=w_vtheory.chunk(2,0)
k,_ =i.chunk(2,1)
_,l=j.chunk(2,1)
theory=torch.cat((k,l),1)
theory=theory.transpose(0,1)
print(theory)

print()

loss = out.mean()
loss.backward()

wv_gradient = attention.w_v.weight.grad
print(wv_gradient)
print()
print('W^V error is {}'.format((theory-wv_gradient).short()))




