import numpy as np 
import neural as nn
import cv2 as cv
import sys
s0 = [1,1,1,1,0,1,1,0,1,1,0,1,1,1,1]
s1 = [0,1,0,0,1,0,0,1,0,0,1,0,0,1,0]
s2 = [1,1,1,0,0,1,1,1,1,1,0,0,1,1,1]
s3 = [1,1,1,0,0,1,1,1,1,0,0,1,1,1,1]
s4 = [1,0,1,1,0,1,1,1,1,0,0,1,0,0,1]
s5 = [1,1,1,1,0,0,1,1,1,0,0,1,1,1,1]
s6 = [1,1,1,1,0,0,1,1,1,1,0,1,1,1,1]
s7 = [1,1,1,0,0,1,0,0,1,0,0,1,0,0,1]
s8 = [1,1,1,1,0,1,1,1,1,1,0,1,1,1,1]
s9 = [1,1,1,1,0,1,1,1,1,0,0,1,0,0,1]

in_set = np.array([s0,s1,s2,s3,s4,s5,s6,s7,s8,s9])

s = [15,10,1]
n = nn.neural(s)
n.load_weights("number1.dat","number2.dat")

I = cv.imread(sys.argv[1],0)/255.
#in_data = [sum(sum(I[(i/3)*10:(i/3)*10+10,(i%3)*16:(i%3)*16+16]))/160. for i in range(15)]
in_data = np.array([[1.0*(1+np.round(sum(sum(I[(i/3)*10:(i/3)*10+10,(i%3)*16:(i%3)*16+16]))/170))%2 for i in range(15)]])
n.forward(in_data)
print in_data
print np.round(9.0*n.o)