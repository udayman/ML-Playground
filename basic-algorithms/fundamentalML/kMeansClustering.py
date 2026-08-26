class KMeans:
    def __init__(self, k=3, maxIters=100):
        self.k = k
        self.maxIters = maxIters
    
    def fit(self, X):
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]

        for _ in range(self.maxIters):
            clusters = [[] for _ in range(self.k)]
            for idx, x in enumerate(X):
                closestIdx = np.argmin([np.linalg.norm(x - c) for c in self.centroids])
                clusters[closestIdx].append(x)
            
            prevCentroids = self.centroids.copy()
            for i in range(self.k):
                if clusters[i]:
                    self.centroids[i] = np.mean(clusters[i], axis=0)
            
            if np.all(prevCentroids == self.centroids):
                break