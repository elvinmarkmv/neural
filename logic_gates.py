import numpy as np
import neural as nn
#Sample with AND logic gate:
s = [2,3,1]
in_set = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
out_set = np.array([[0.],[0.],[0.],[1.]])

n = nn.neural(s)
n.train(in_set,out_set,3000)
n.forward(in_set)
print n.o