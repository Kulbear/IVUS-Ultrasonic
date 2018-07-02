import tensorflow as tf
import tensorlayer as tl
import numpy as np


M_tree = np.array([[0., 1., 1., 1., 1.],
                   [1., 0., 0.6, 0.2, 0.5],
                   [1., 0.6, 0., 0.6, 0.7],
                   [1., 0.2, 0.6, 0., 0.5],
                   [1., 0.5, 0.7, 0.5, 0.]], dtype=np.float64)


def dice_loss(logits, y):
    return 1 - tl.cost.dice_coe(logits, y, axis=[1, 2], loss_type='sorensen')

def jacc_loss(logits, y):
    return 1 - tl.cost.dice_coe(logits, y, axis=[1, 2], loss_type='sorensen')

def sigmoid_cross_entropy(logits, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)

def softmax_cross_entropy(logits, y):
    return tf.nn.softmax_cross_entropy_v2(logits=logits, labels=y)

def wasserstein_disagreement_map(prediction, ground_truth, M):
    """
    Function to calculate the pixel-wise Wasserstein distance between the
    flattened pred_proba and the flattened labels (ground_truth) with respect
    to the distance matrix on the label space M.

    :param prediction: the logits after softmax
    :param ground_truth: segmentation ground_truth
    :param M: distance matrix on the label space
    :return: the pixelwise distance map (wass_dis_map)
    """
    # pixel-wise Wassertein distance (W) between flat_pred_proba and flat_labels
    # wrt the distance matrix on the label space M
    n_classes = prediction.get_shape()[1].value
    unstack_labels = tf.unstack(ground_truth, axis=-1)
    unstack_labels = tf.cast(unstack_labels, dtype=tf.float64)
    unstack_pred = tf.unstack(prediction, axis=-1)
    unstack_pred = tf.cast(unstack_pred, dtype=tf.float64)

    # W is a weighting sum of all pairwise correlations (pred_ci x labels_cj)
    pairwise_correlations = []
    for i in range(n_classes):
        for j in range(n_classes):
            pairwise_correlations.append(
                M[i, j] * tf.multiply(unstack_pred[i], unstack_labels[j]))
    wass_dis_map = tf.add_n(pairwise_correlations)
    return wass_dis_map


def generalised_wasserstein_dice_loss(prediction,
                                      ground_truth,
                                      weight_map=None):
    """
    Function to calculate the Generalised Wasserstein Dice Loss defined in
    Fidon, L. et. al. (2017) Generalised Wasserstein Dice Score for Imbalanced
    Multi-class Segmentation using Holistic Convolutional Networks.
    MICCAI 2017 (BrainLes)
    :param prediction: the logits (before softmax)
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    # apply softmax to pred scores
    ground_truth = tf.cast(ground_truth, dtype=tf.int64)
    pred_proba = tf.nn.softmax(tf.cast(prediction, dtype=tf.float64))
    n_classes = prediction.get_shape()[1].value
    n_voxels = prediction.get_shape()[0].value
    ids = tf.constant(np.arange(n_voxels), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)

    one_hot = tf.SparseTensor(indices=ids,
                              values=tf.ones([n_voxels], dtype=tf.float32),
                              dense_shape=[n_voxels, n_classes])
    one_hot = tf.sparse_tensor_to_dense(one_hot)
    # M = tf.cast(M, dtype=tf.float64)
    # compute disagreement map (delta)
    M = M_tree
    # print("M shape is ", M.shape, pred_proba, one_hot)
    delta = wasserstein_disagreement_map(pred_proba, one_hot, M)
    # compute generalisation of all error for multi-class seg
    all_error = tf.reduce_sum(delta)
    # compute generalisation of true positives for multi-class seg
    one_hot = tf.cast(one_hot, dtype=tf.float64)
    true_pos = tf.reduce_sum(
        tf.multiply(tf.constant(M[0, :n_classes], dtype=tf.float64), one_hot),
        axis=1)
    true_pos = tf.reduce_sum(tf.multiply(true_pos, 1. - delta), axis=0)
    WGDL = 1. - (2. * true_pos) / (2. * true_pos + all_error)
    return tf.cast(WGDL,dtype=tf.float32)
