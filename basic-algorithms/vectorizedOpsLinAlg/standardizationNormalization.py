import numpy as np

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    stdReplaced = np.where(std==0, 1, std)

    return (X - mean)/stdReplaced

data = np.array([[10,2], [20,2], [30,2]])
print("Standardizing data")
print(standardize(data))

def minMaxScale(X, featureRange = (0,1)):
    minX = np.min(X, axis=0)
    maxX = np.max(X, axis=0)

    diff = maxX - minX
    diffReplaced = np.where(diff == 0, 1, diff)

    normX = (X - minX) / diffReplaced
    minR, maxR = featureRange
    return normX * (maxR - minR) + minR

print(minMaxScale(data))
    
