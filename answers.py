import numpy as np
import torch
import pandas as pd
from tabulate import tabulate

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

    h.show()

    o = [a1, sg_z_1, a2, sg_z_2, a3, h, sg_loss, sg_w_1, sg_b_1, sg_w_2, sg_b_2, sg_w_3, sg_b_3]

    variables = pd.DataFrame({
        'Variables': [
            'Pre-Activation Couche 1:', 'Activation Couche 1:', 'Pre-Activation Couche 2:',
            'Activation Couche 2:', 'Pre-Activation Couche 3:', 'Activation Couche 3:',
            'Fonction de perte:', 'Poids Couche 1:', 'Bias Couche 1:',
            'Poids Couche 2:', 'Bias Couche 2:', 'Poids Couche 3:',
            'Bias Couche 3:'
        ],
        'Valeurs': [v.data for v in o],
        'Gradients': [v.grad for v in o],
        # 'Valeurs apres backprop': [v.data - lr * v.grad for v in o]
    })

    print(tabulate(variables.to_records(), headers = 'keys', tablefmt = 'psql'))

if __name__ == '__main__':
    forward()
