import numpy as np
import torch
import torch.nn.functional as F

from scratch_grad import Variable

def forward():
    w_1_data = np.array([[1, -1, -2], [-.6, .2, -.4]]).astype(np.float32)
    w_2_data = np.array([[.3, .8], [-1.5, -.8]]).astype(np.float32)
    w_3_data = np.array([[-.5, 1.5]]).astype(np.float32)
    b_1_data = np.array([[.5], [2.8]]).astype(np.float32)
    b_2_data = np.array([[-.1], [0.4]]).astype(np.float32)
    b_3_data = np.array([0.5]).astype(np.float32)
    lr = 0.05

    x = torch.tensor([[2], [1], [1]], dtype=torch.float32)
    y = torch.tensor([1], dtype=torch.float32)

    w_1 = torch.tensor(w_1_data, requires_grad=True)
    b_1 = torch.tensor(b_1_data, requires_grad=True)
    w_2 = torch.tensor(w_2_data, requires_grad=True)
    b_2 = torch.tensor(b_2_data, requires_grad=True)
    w_3 = torch.tensor(w_3_data, requires_grad=True)
    b_3 = torch.tensor(b_3_data, requires_grad=True)

    a_1 = (w_1 @ x + b_1)
    z_1 = F.relu(a_1)
    a_1.retain_grad()
    z_1.retain_grad()
    a_2 = (w_2 @ z_1 + b_2)
    a_2.retain_grad()
    z_2 = F.relu(a_2)
    z_2.retain_grad()
    z_3 = (w_3 @ z_2 + b_3)
    z_3.retain_grad()
    softmax_out = F.sigmoid(z_3)
    softmax_out.retain_grad()
    loss = (-torch.log(softmax_out))
    loss.retain_grad()

    o = [a_1, z_1, a_2, z_2, z_3, softmax_out, w_1, w_2, w_3, b_1, b_2, b_3, loss]
    for i, v in zip(range(len(o)), o):
        print(f"Variable {v}:")
        print(f"Gradient {v.grad}:")
        print("-------")
    print("-------------------------------------------")

    loss.backward()

    for i, v in zip(range(len(o)), o):
        print(f"Variable {v}:")
        print(f"Gradient {v.grad}:")
        print(f"Valeur apres {v - (lr * v.grad)}:")
        print("-------")

    print("-------------------------------------------")

    # scratch_grad
    sg_w_1 = Variable(w_1_data, name='w1')
    sg_b_1 = Variable(b_1_data, name='b1')
    sg_w_2 = Variable(w_2_data, name='w2')
    sg_b_2 = Variable(b_2_data, name='b2')
    sg_w_3 = Variable(w_3_data, name='w3')
    sg_b_3 = Variable(b_3_data, name='b3')

    sg_x = Variable(x.numpy(), name='x')

    a1 = sg_w_1 @ sg_x  + sg_b_1
    sg_z_1 = a1.relu()
    a2 = sg_w_2 @ sg_z_1 + sg_b_2
    sg_z_2 = a2.relu()
    a3 = sg_w_3 @ sg_z_2 + sg_b_3

    h = a3.sigmoid()

    sg_loss = (-Variable(y.numpy())) * h.log()

    sg_loss.backward()

    sg_loss.show()
    # h.show()

    o = [a1, sg_z_1, a2, sg_z_2, a3, sg_w_1, sg_w_2, sg_w_3, sg_b_1, sg_b_2, sg_b_3, h, sg_loss]
    for i, v in zip(range(len(o)), o):
        print(f"Variable {v}:")
        print(f"Valeur avant {v.data}:")
        print(f"Gradient {v.grad}:")
        print(f"Valeur apres {v.data - lr * v.grad}:")
        print("-------")

if __name__ == '__main__':
    forward()
