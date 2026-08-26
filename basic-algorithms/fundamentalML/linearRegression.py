import numpy as np
class SGDLinearRegression:
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0
    
    def fit(self, X, y):
        nSamples, nFeatures = X.shape
        self.weights = np.zeros(nFeatures)

        for _ in range(self.epochs):
            for i in range(nSamples):
                # Forward pass to get predictions
                yPred = np.dot(X[i], self.weights) + self.bias

                # take derivative of mse(Y) with respect to w
                dw = 2 * X[i] * (yPred - y[i])
                db = 2 * (yPred - y[i])

                self.weights -= self.lr * dw 
                self.bias -= self.lr * db 
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias 

# Synthetic Data of square footage as 1st feature, house age as second
X = np.array([
    [1.5, 5],
    [2.0, 10],
    [1.2, 2],
    [3.5, 15],
    [2.2, 7]
])

# estimating price in 100,000s factor
y = np.array([3.1, 3.9, 2.6, 6.2, 4.3])

# intiialize and then train
model = SGDLinearRegression(lr=0.01, epochs=200)
model.fit(X, y)

# Make a prediction to predict a house with 2.5 scale sqft and 8 yrs old
newHouse = np.array([2.5, 8])
prediction = model.predict(newHouse)

print(f"Weights:{model.weights}")
print(f"Bias:{model.bias:.4f}")
print(f"Predicted Price for new house: ${prediction * 100:.2f}k")