import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

def normalize_feature(points):
    points = points / points.max()
    return points


def get_columns(points, feature_columns):
    return points[feature_columns]

def DBSCAN_impl(points, feature_columns):

    points = points[feature_columns].dropna()
    processed_points = get_columns(points, feature_columns)
    # processed_points = normalize_feature(processed_points)
    print processed_points.to_string()
    X = processed_points.values.tolist()

    X = np.array(X)
    # X = X.fit_transform(X)
    # #############################################################################
    # When ground truth is not available, label the data with clustering algorithm
    # Compute DBSCAN
    db = DBSCAN(eps=0.03, min_samples=20).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))
    unique, counts = np.unique(labels, return_counts=True)
    print dict(zip(unique, counts))
    # #############################################################################
    # Plot result
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        ax.scatter(xs=xy[:, 0], ys=xy[:, 1], zs=xy[:, 2], marker = 'o', color=tuple(col),
                 s=6)

        xy = X[class_member_mask & ~core_samples_mask]
        ax.scatter(xs=xy[:, 0], ys=xy[:, 1], zs=xy[:, 2], marker = 'o', color=tuple(col),
                 s=6)
    ax.set_xlabel(feature_columns[0])
    ax.set_ylabel(feature_columns[1])
    ax.set_zlabel(feature_columns[2])
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

    classification = {}

    # Recognize the clusters that are positives.
    # To find the labels, find a point which we know the label.
    classified_cluster_n = set()
    for i in range(0, len(X)):
        if len(classified_cluster_n) == len(unique_labels):
            break
        x = X[i]
        # Extract verified labels
        if x[0] - x[1] < - 0.6:
            classification[labels[i]] = "P"
            classified_cluster_n.add(labels[i])
        elif x[0] - x[1] < - 0.4 and x[0] < 0.05:
            classification[labels[i]] = "P"
            classified_cluster_n.add(labels[i])
        elif abs(x[0] - x[1]) > 0 and x[0] > 0.5 :
            classification[labels[i]] = "N"
            classified_cluster_n.add(labels[i])
        elif abs(x[0] - x[1]) < 0.2:
            classification[labels[i]] = "U"
            classified_cluster_n.add(labels[i])

    print classification



    # Create a column in the data frame
    label_column = np.array([classification[labels[i]] for i in range(0, len(X))])
    cluster_column = np.array([labels[i] for i in range(0, len(X))])
    points["cluster"] = pd.Series(cluster_column, index=points.index)
    points["label"] = pd.Series(label_column, index=points.index)
    points = points[points.cluster != -1]

    print "Labeling done."
    return points, n_clusters_

def kmeans(points, clusters_n, iteration_n):

    points = get_columns(points, ["c_c_dpr_lr_9000", "w_c_dpr_lr_0_9000"])
    print points.to_string()
    iteration_n = 2000
    points = tf.constant(points)


    # DEBUG
    # points_n = 300
    # points = tf.constant(np.random.uniform(0, 10, (points_n, 2)))


    centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0, 0], [clusters_n, -1]))

    points_expanded = tf.expand_dims(points, 0)
    centroids_expanded = tf.expand_dims(centroids, 1)

    distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
    assignments = tf.argmin(distances, 0)

    means = []
    for c in range(clusters_n):
        means.append(tf.reduce_mean(
            tf.gather(points,
                      tf.reshape(
                          tf.where(
                              tf.equal(assignments, c)
                          ), [1, -1])
                      ), reduction_indices=[1]))

    new_centroids = tf.concat(means, 0)

    update_centroids = tf.assign(centroids, new_centroids)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for step in range(iteration_n):
            [_, centroid_values, points_values, assignment_values] = sess.run(
                [update_centroids, centroids, points, assignments])

        print "centroids" , centroid_values
        # Print number of points in cluster
        unique, counts = np.unique(assignment_values, return_counts=True)
        print dict(zip(unique, counts))

    plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.5)
    plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)
    plt.xlabel("c_c_dpr_lr_9000")
    plt.ylabel("w_c_dpr_lr_0_9000")
    plt.show()


