# neural
Personal backup of the simple neural network coded in python with some examples on how to use it
# Notes:
ann.py: simple file implementing an artificial neural network (3 layers only) with a backpropagation method (for training) the ANN is tested with a simple logic gate (AND)
neural.py: module containing class "neural" which is ann.py converted into a class for generalation purposes.
other files: examples using neural.py
# Cautions: (for future me XD)
In implementing backpropagation method, the error to be propagated is not just the difference between the the predicted output and the actual output (po - ao) but (po - ao) times po(1-po).
Also during the initiation of weigths is better to initiate them randomly because if not, if we just put ones or zeros, the recursion will converge to a different output. This may be caused by the presence of different minimums, so the when all the values are ones, for example, the recursion will converge to the minimum closest to (1,1,1,.....,1) which may not be the minimum we want.
