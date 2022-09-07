
import torch
from multi_head_test import MultiHeadAttention




x =torch.rand(1,5,10)
attention = MultiHeadAttention(d_model=10, n_head=2)

out,gradient_wo,att1,score,A = attention(q=x, k=x, v=x)
'''
out 就是正常多头注意力的输出，gradient_wo是Wo的梯度,att1是未经过拼接的attention结果是一个四维张量,score是经过拼接的attention结果在示例代码中是一个二维张量因为batch_size=1,A
就是未经过softmax的attention计算结果，跟推导中的定义一致

out is the output of the Multi_head attention,gradient_wo is the gradient of W^O,att1 is a 4-dim tensor before concat, the score is a 2-dim tensor 
after concat, this is because our batch_size  is 1, and A is the attention layer output before softmax, which is the same definition as my derived.

'''
# print(A)
x=x.squeeze()
print('x is {}'.format(x))
I=torch.block_diag(torch.ones(5,5),torch.ones(5,5))
Y=1/(A.exp()@I+1e-15) # add a small positive to prevent divid zero
# print(attention.w_concat.weight)
vp = torch.block_diag(x@attention.w_v.weight.transpose(0,1)[:,0:5],x@attention.w_v.weight.transpose(0,1)[:,5:10])
# print(vp)
dev_A = ((torch.ones(5,10)/50)@attention.w_concat.weight@vp.transpose(0,1))*score-((((torch.ones(5,10)/50)@attention.w_concat.weight@ \
                                                                                    vp.transpose(0,1))*score*Y)@I.transpose(0,1))*A.exp()

# print(dev_A)
# print(attention.w_k.weight.transpose(0,1)[:,0:5])
ph1 = torch.cat((torch.eye(5,5),torch.zeros(5,5)),1)
ph2 = torch.cat((torch.zeros(5,5),torch.eye(5,5)),1)
w_q1 = (1/torch.sqrt(torch.tensor(5)))*x.transpose(0,1)@dev_A@ph1.transpose(0,1)@x@attention.w_k.weight.transpose(0,1)[:,0:5]
w_q2 = (1/torch.sqrt(torch.tensor(5)))*x.transpose(0,1)@dev_A@ph2.transpose(0,1)@x@attention.w_k.weight.transpose(0,1)[:,5:10]
w_q = torch.cat((w_q1,w_q2),1)
print('w_q_theory is {}'.format(w_q))
# print(out.mean())
print()
loss = out.mean()
loss.backward()
# loss=criterion(out, trg)
wq_gradient = attention.w_q.weight.grad
w_q_true = wq_gradient.transpose(0,1)
print('w_q_true is {}'.format(w_q_true))
print()
print('W^Q error is {}'.format((w_q-w_q_true).short()))




