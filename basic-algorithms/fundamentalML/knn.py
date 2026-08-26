class KNN:
    def __init__(self, k=3):
        self.k = k 
    
    def fit(self, X, y):
        self.xTrain = X 
        self.yTrain = Y
    
    def predict(self, X):
        return np.array([self._predict(x) for x in X])
    
    def _predict(self, x):
        # compute Euclidean distances
        distances = [np.sqrt(np.sum((x - xtrain)**2)) for xtrain in self.xTrain]

        # get indices of k nearest neighbors
        kIndices = np.argsort(distances)[:self.k]

        # Extract labels and return most common (Majority vote)
        kNearestLabels = [self.yTrain[i] for i in kIndices]
        return max(set(kNearestLabels), key=kNearestLabels.count)