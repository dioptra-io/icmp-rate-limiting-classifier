import tensorflow as tf
import random

def bucket_mean(data, bucket_ids, num_buckets):
    total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
    count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
    return total / count



def kmeans(data, data_size, cluster_num, max_iter):

    K = cluster_num
    MAX_ITERS = max_iter
    # Take random centroid from the data.
    centroids_indexes = []
    for i in range(0, K):
        centroids_indexes.append(random.randint(0, data_size))

    centroids = tf.Variable([data[centroids_indexes[i]] for i in range(0, len(centroids_indexes))])

    init_op = tf.initialize_all_variables()

    # run the graph
    with tf.Session() as sess:
        sess.run(init_op)  # execute init_op
        # print the random values that we sample
        print (sess.run(centroids))

    # Replicate to N copies of each centroid and K copies of each
    # point, then subtract and compute the sum of squared distances.
    # rep_centroids = tf.reshape(tf.tile(centroids, [N, 1]), [N, K, 2])
    # rep_points = tf.reshape(tf.tile(points, [1, K]), [N, K, 2])
    # sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids),
    #                             reduction_indices=2)
    #
    # # Use argmin to select the lowest-distance point
    # best_centroids = tf.argmin(sum_squares, 1)
    # did_assignments_change = tf.reduce_any(tf.not_equal(best_centroids,
    #                                                     cluster_assignments))
    #
    #
    #
    # means = bucket_mean(points, best_centroids, K)
    #
    # # Do not write to the assigned clusters variable until after
    # # computing whether the assignments have changed - hence with_dependencies
    # with tf.control_dependencies([did_assignments_change]):
    #     do_updates = tf.group(
    #         centroids.assign(means),
    #         cluster_assignments.assign(best_centroids))
    #
    # init = tf.initialize_all_variables()
    #
    # sess = tf.Session()
    # sess.run(init)
    #
    # changed = True
    # iters = 0
    #
    # while changed and iters < MAX_ITERS:
    #     iters += 1
    #     [changed, _] = sess.run([did_assignments_change, do_updates])
    #
    # [centers, assignments] = sess.run([centroids, cluster_assignments])
    # end = time.time()
    # print ("Found in %.2f seconds" % (end-start)), iters, "iterations"
    # print "Centroids:"
    # print centers
    # print "Cluster assignments:", assignments