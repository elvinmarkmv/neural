import numpy as np 
import neural as nn
import matplotlib.pyplot as plt 

x = np.linspace(0,5,20)
y = x*x*x + x*x + 2
y = y + 0.2*np.random.normal(size=(20))

x_max = np.max(x)
y_max = np.max(y)
in_set = np.array([x/x_max]).T
out_set = np.array([y/y_max]).T
s = [1,10,1]
n = nn.neural(s)
n.train(in_set,out_set,2000)
n.forward(in_set)
y_p = n.o
plt.figure(1)
plt.plot(x,y,"*")
plt.plot(x,y_max*y_p.T[0])
plt.show()