import numpy as np 
import neural as nn
import cv2 as cv
import sys

in_set = [] 
Nx = 10
Ny = 10
hx = 50/Nx
hy = 50/Ny
"""
for k in range(1,6):
	for j in range(1,5):
		print str(k)+"-"+str(j)+".png"
		I = cv.imread(str(k)+"-"+str(j)+".png",0)/255.
		in_set.append(np.array([sum(sum(I[(i/hx)*Nx:(i/hx)*Nx+Nx,(i%hy)*Ny:(i%hy)*Ny+Ny]))/(Nx*Ny) for i in range(25)]))

in_set = np.array(in_set)
"""

s = [25,10,1]
n = nn.neural(s)
n.load_weights("number1.dat","number2.dat")

I = cv.imread(sys.argv[1],0)/255.
print sys.argv[1]
in_data = np.array([np.array([sum(sum(I[(i/hx)*Nx:(i/hx)*Nx+Nx,(i%hy)*Ny:(i%hy)*Ny+Ny]))/(Nx*Ny) for i in range(25)])])

n.forward(in_data)
print np.round(9.0*n.o)