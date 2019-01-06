import numpy as np
#activation function - sigmoid function
def sigmoid(x):
	return 1./(1 + np.exp(-x))
#init
s = [2,3,1]
w = [np.random.randn(j,i) for i,j in zip(s[:-1],s[1:])]

#train_set
in_set = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
out_set = np.array([[0.],[0.],[0.],[1.]])

#get_output
out = in_set.T
for q in w:
	out = sigmoid(q.dot(out))
print out.T

#train
i = 0
N =3000
while i<N:
	o0 = in_set
	o1 = sigmoid((w[0].dot(o0.T)).T)
	o2 = sigmoid((w[1].dot(o1.T)).T)
	e2 = (o2 - out_set)
	delta2 = e2*o2*(1-o2)
	e1 = delta2.dot(w[1])
	delta1 = e1*o1*(1-o1)
	w[0] = w[0] - delta1.T.dot(o0)
	w[1] = w[1] - delta2.T.dot(o1)
	i = i + 1
out = in_set.T
for q in w:
	out = sigmoid(q.dot(out))
print out.T