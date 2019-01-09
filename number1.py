import numpy as np 
import neural as nn
import cv2 as cv
#load_samples
in_set = [] 
out_set = []
Nx = 10
Ny = 10
hx = 50/Nx
hy = 50/Ny
for k in range(1,6):
	for j in range(1,5):
		print str(k)+"-"+str(j)+".png"
		I = cv.imread(str(k)+"-"+str(j)+".png",0)/255.
		in_set.append(np.array([sum(sum(I[(i/hx)*Nx:(i/hx)*Nx+Nx,(i%hy)*Ny:(i%hy)*Ny+Ny]))/(Nx*Ny) for i in range(25)]))
		out_set.append([k])

in_set = np.array(in_set)
out_set = np.array(out_set)/9.

s = [25,10,1]
n = nn.neural(s)
n.train(in_set,out_set,2000)
n.save_weights("number1.dat","number2.dat")
n.forward(in_set)
print np.round(9.*n.o)