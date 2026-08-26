import numpy as np

#Showcasing Broadcasting
# weighted sums
X = np.array([[1,2],[3,4],[5,6]]) # (3,2) array
w = np.array([0.1,0.2]) #(2,) array

# produce a (3,2) array
weightedFeatures = X * w
# todo, use the einsum here
weightedFeaturesEinsum = np.einsum('ij,j->ij',X,w)

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
def softmax(X):
    # take max of elements for normalization
    # Keep dims to preserve (N,1)
    rowMax = np.max(X, axis=1, keepdims=True)

    # subtract the max from each element
    # broadcasting of (N,D) - (N,1) works here
    subtractedMax = X - rowMax

    # take exponent of each element, by passing in np.exp function
    exps = np.exp(subtractedMax)

    # divide by the row sum, which we compute using np.sum and keep dims for (N,1)
    rowSum = np.sum(exps, axis=1, keepdims=True)
    return exps / rowSum

print("softmax here")
print(softmax(output))

