import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


X, _ = make_blobs(n_samples=500, centers=3, n_features=2, random_state=42)

def initialize_centroids(X, K):
    centroids = np.zeros((K, X.shape[1]))
    for i in range(X.shape[1]):
        min_val, max_val = np.min(X[:, i]), np.max(X[:, i])
        centroids[:, i] = np.linspace(min_val, max_val, K)
    return centroids

def assign_clusters(X, centroids):
    distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)


def compute_centroids(X, labels, K):
    return np.array([X[labels == k].mean(axis=0) for k in range(K)])


def plot_clusters(X, centroids, labels=None, iteration=None):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
    if iteration is not None:
        plt.title(f"Iteration {iteration}")
    else:
        plt.title("Initial Centroids")
    plt.show()

def kmeans(X, K, max_iters=100):
    centroids = initialize_centroids(X, K)

    plot_clusters(X, centroids)
    
    for iteration in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = compute_centroids(X, labels, K)
        plot_clusters(X, new_centroids, labels, iteration + 1)

        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, labels

K = 3
final_centroids, final_labels = kmeans(X, K)
plt.show()
