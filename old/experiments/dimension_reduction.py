import numpy as np
from sklearn.decomposition import PCA


def main():
    # Przykładowe wektory
    x1 = np.random.rand(1000)
    x2 = np.random.rand(1000)

    X = np.vstack([x1, x2])

    n_samples, n_features = X.shape
    print(f"n_samples: {n_samples}, n_features: {n_features}")

    k = min(n_samples, n_features, 10)
    pca = PCA(n_components=k)
    X_reduced = pca.fit_transform(X)

    x1_reduced = X_reduced[0]
    x2_reduced = X_reduced[1]
    distance_reduced = np.linalg.norm(x1_reduced - x2_reduced)
    distance_original = np.linalg.norm(x1 - x2)

    print(f"Reduced distance: {distance_reduced} for vectors of shape {x1_reduced.shape}")
    print(f"Original distance: {distance_original} for vectors of shape {x1.shape}")


if __name__ == '__main__':
    main()
