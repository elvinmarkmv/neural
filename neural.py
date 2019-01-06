import numpy as np
#This class is just for a 3 layer neural network (since most of the problems can be solved with this kind of neural networks): 
#input_layer - hidden_layer - output_layer
class neural():
	def __init__(self,s):
		self.s = s
		self.w = [np.random.randn(j,i) for i,j in zip(s[:-1],s[1:])]
	def sigmoid(self,x):
		return 1./(1 + np.exp(-x))
	def forward(self,in_data):
		out = in_data.T
		for q in self.w:
			out = self.sigmoid(q.dot(out))
		self.o = out.T
	def backward(self,in_set,out_set):
		o0 = in_set
		o1 = self.sigmoid((self.w[0].dot(o0.T)).T)
		o2 = self.sigmoid((self.w[1].dot(o1.T)).T)
		e2 = (o2 - out_set)
		delta2 = e2*o2*(1-o2)
		e1 = delta2.dot(self.w[1])
		delta1 = e1*o1*(1-o1)
		self.w[0] = self.w[0] - delta1.T.dot(o0)
		self.w[1] = self.w[1] - delta2.T.dot(o1)
	def train(self,in_set,out_set,N):
		i = 0
		while i<N:
			self.backward(in_set,out_set)
			i = i + 1
	def load_weights(self,dir1,dir2):
		self.w = [np.reshape(np.loadtxt(dir1),(self.s[1],self.s[0])),np.reshape(np.loadtxt(dir2),(self.s[2],self.s[1]))]
	def save_weights(self,dir1,dir2):
		np.savetxt(dir1,self.w[0])
		np.savetxt(dir2,self.w[1])
