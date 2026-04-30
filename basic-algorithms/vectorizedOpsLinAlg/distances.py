import numpy as np

def euclideanDistances(p, M):
    return np.linalg.norm(M - p, axis=1)

def cosineSimilarity(A, B):
    Anorm = A / np.linalg.norm(A)

    Bnorm = B / np.linalg.norm(B, axis=1, keepdims = True)

    return np.dot(Bnorm, Anorm)

def manhattanDistance(p, M):
    return np.sum(np.abs(M - p), axis=1)