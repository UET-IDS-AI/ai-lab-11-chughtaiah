"""
AI_stats_lab.py

Instructor Solution
Lab: Unsupervised Learning and K-Means Clustering
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


# ============================================================
# Question 1: Unlabeled Data and K-Means Clustering
# ============================================================

def load_iris_unlabeled(feature_indices=(0, 1)):
    """
    Load the Iris dataset without labels.
    """
    iris = load_iris()

    X = iris.data[:, feature_indices]
    feature_names = [iris.feature_names[i] for i in feature_indices]

    return {
        "X": X,
        "feature_names": feature_names
    }


def standardize_features(X):
    """
    Standardize features to zero mean and unit variance.
    """
    X = np.array(X, dtype=float)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0

    X_scaled = (X - mean) / std_safe

    return {
        "X_scaled": X_scaled,
        "mean": mean,
        "std": std_safe
    }


def fit_kmeans(X, K, random_state=0, n_init=10):
    """
    Fit K-Means clustering on data X.
    """
    model = KMeans(
        n_clusters=K,
        random_state=random_state,
        n_init=n_init
    )

    model.fit(X)

    return {
        "centroids": model.cluster_centers_,
        "labels": model.labels_,
        "objective": model.inertia_,
        "n_iter": model.n_iter_
    }


def compute_kmeans_objective(X, centroids, labels):
    """
    Compute the K-Means objective manually.
    """
    X = np.array(X, dtype=float)
    centroids = np.array(centroids, dtype=float)
    labels = np.array(labels)

    objective = 0.0

    for i in range(len(X)):
        assigned_centroid = centroids[labels[i]]
        diff = X[i] - assigned_centroid
        objective += np.sum(diff ** 2)

    return float(objective)


# ============================================================
# Question 2: Choosing K, Underfitting/Overfitting, and Outliers
# ============================================================

def evaluate_k_values(X, k_values, random_state=0, n_init=10):
    """
    Run K-Means for multiple values of K.
    """
    objectives = []

    for K in k_values:
        result = fit_kmeans(
            X,
            K=K,
            random_state=random_state,
            n_init=n_init
        )

        objectives.append(result["objective"])

    relative_improvements = []

    for i in range(len(objectives)):
        if i == 0:
            relative_improvements.append(0.0)
        else:
            previous_objective = objectives[i - 1]
            current_objective = objectives[i]

            if previous_objective == 0:
                improvement = 0.0
            else:
                improvement = (
                    previous_objective - current_objective
                ) / previous_objective

            relative_improvements.append(float(improvement))

    return {
        "k_values": k_values,
        "objectives": objectives,
        "relative_improvements": relative_improvements
    }


def choose_elbow_k(k_values, objectives):
    """
    Choose K using maximum-distance-to-line elbow heuristic.
    """
    if len(k_values) < 3:
        return k_values[0]

    k_values_array = np.array(k_values, dtype=float)
    objectives_array = np.array(objectives, dtype=float)

    points = np.column_stack((k_values_array, objectives_array))

    first_point = points[0]
    last_point = points[-1]

    line_vector = last_point - first_point
    line_length = np.linalg.norm(line_vector)

    if line_length == 0:
        return k_values[0]

    distances = []

    for point in points:
        point_vector = point - first_point

        distance = np.abs(
            np.cross(line_vector, point_vector)
        ) / line_length

        distances.append(distance)

    best_index = int(np.argmax(distances))

    return k_values[best_index]


def cluster_size_summary(labels, K):
    """
    Count how many data points belong to each cluster.
    """
    labels = np.array(labels)

    summary = {}

    for cluster_index in range(K):
        summary[cluster_index] = int(np.sum(labels == cluster_index))

    return summary


def identify_outliers_by_distance(X, centroids, labels, top_n=5):
    """
    Identify possible outliers based on distance from assigned centroid.
    """
    X = np.array(X, dtype=float)
    centroids = np.array(centroids, dtype=float)
    labels = np.array(labels)

    distances = []

    for i in range(len(X)):
        assigned_centroid = centroids[labels[i]]
        diff = X[i] - assigned_centroid
        squared_distance = np.sum(diff ** 2)
        distances.append(squared_distance)

    distances = np.array(distances)

    sorted_indices = np.argsort(distances)[::-1]

    top_indices = sorted_indices[:top_n]
    top_distances = distances[top_indices]

    return {
        "indices": top_indices,
        "distances": top_distances
    }


def diagnose_clustering_fit(K, elbow_k):
    """
    Diagnose whether K is underfitting, good fit, or overfitting.
    """
    if K < elbow_k:
        return "underfitting"

    if K == elbow_k:
        return "good_fit"

    return "overfitting"


# ============================================================
# Question 3: Visualization
# ============================================================

def plot_unlabeled_data(X, feature_names=None, title="Unlabeled Data"):
    """
    Visualize unlabeled 2D data.
    """
    X = np.array(X)

    fig, ax = plt.subplots()

    ax.scatter(X[:, 0], X[:, 1])

    if feature_names is None:
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
    else:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])

    ax.set_title(title)

    return fig, ax


def plot_kmeans_clusters(X, labels, centroids, feature_names=None, title="K-Means Clusters"):
    """
    Visualize K-Means clustering results.
    """
    X = np.array(X)
    labels = np.array(labels)
    centroids = np.array(centroids)

    fig, ax = plt.subplots()

    ax.scatter(X[:, 0], X[:, 1], c=labels)

    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="D",
        s=120,
        edgecolors="black"
    )

    if feature_names is None:
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
    else:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])

    ax.set_title(title)

    return fig, ax


def plot_elbow_curve(k_values, objectives, title="Elbow Method"):
    """
    Plot K-Means objective values versus K.
    """
    fig, ax = plt.subplots()

    ax.plot(k_values, objectives, marker="o")

    ax.set_xlabel("Number of clusters K")
    ax.set_ylabel("Objective value")
    ax.set_title(title)

    return fig, ax


if __name__ == "__main__":
    print("Instructor solution loaded.")
