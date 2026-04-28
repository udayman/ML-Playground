import numpy as np

#Showcasing Broadcasting
# weighted sums
X = np.array([[1,2],[3,4],[5,6]]) # (3,2) array
w = np.aray([0.1,0.2]) #(2,) array

# produce a (3,2) array
weightedFeatures = X * w
# todo, use the einsum here
weightedFeaturesEinsum = np.einsum('ij,j->ij')

# produce an array summing over all the cols to form a (3,) array
yPred = np.sum(weightedFeatures, axis=1)
print("Showcasing predictions after broadcasting")
print(yPred)


#showcasing outer product
u = np.array([1,2,3]) # (3,) array
v = np.array([4,5]) # (2,) array

#reshape to be a column vector
uCol = u[:, np.newaxis] # turns into a (3,1) array

# Outer product via broadcasting to produce a 3,2 array
outerMatrix = uCol * v
print("Showcasing outer matrix after broadcasting")
print(outerMatrix)

# Batch multiplication
# do operations all over a batch from B x N x M to multiply with M x P weights
A = np.random.rand(10, 3, 4) # batch of 10, 3x4 matrices
W = np.random.rand(4,5) # 4 x 5 weight matrix

# do operation from bij, jk -> bik
output = np.einsum('bij,jk->bik', A, W)
print("Final output")
print(output)

# do softmax here
expVals = np.exp(output)



