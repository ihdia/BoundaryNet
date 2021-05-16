import math
import torch
from sklearn.utils.extmath import cartesian
import numpy as np
from torch.nn import functional as F
import os
import time
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors.kde import KernelDensity
import skimage.io
from torch import nn


torch.set_default_dtype(torch.float32)


def _assert_no_grad(variables):
    for var in variables:
        assert not var.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"


# def cdist(x, y):
#     '''
#     Input: x is a Nxd Tensor
#            y is a Mxd Tensor
#     Output: dist is a NxM matrix where dist[i,j] is the norm
#            between x[i,:] and y[j,:]
#     i.e. dist[i,j] = ||x[i,:]-y[j,:]||
#     '''
#     differences = x.unsqueeze(1) - y.unsqueeze(0)
#     distances = torch.sum(torch.abs(differences), -1)
#     return distances

def cdist(x, y):
    '''
    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances




def averaged_hausdorff_distance(set1, set2, max_ahd=np.inf):
    """
    Compute the Averaged Hausdorff Distance function
     between two unordered sets of points (the function is symmetric).
     Batches are not supported, so squeeze your inputs first!
    :param set1: Array/list where each row/element is an N-dimensional point.
    :param set2: Array/list where each row/element is an N-dimensional point.
    :param max_ahd: Maximum AHD possible to return if any set is empty. Default: inf.
    :return: The Averaged Hausdorff Distance between set1 and set2.
    """

    if len(set1) == 0 or len(set2) == 0:
        return max_ahd

    set1 = np.array(set1)
    set2 = np.array(set2)

    assert set1.ndim == 2, 'got %s' % set1.ndim
    assert set2.ndim == 2, 'got %s' % set2.ndim

    assert set1.shape[1] == set2.shape[1], \
        'The points in both sets must have the same number of dimensions, got %s and %s.'\
        % (set2.shape[1], set2.shape[1])

    d2_matrix = pairwise_distances(set1, set2, metric='euclidean')

    res = np.average(np.min(d2_matrix, axis=0)) + \
        np.average(np.min(d2_matrix, axis=1))

    return res


class AveragedHausdorffLoss(nn.Module):
    def __init__(self):
        super(AveragedHausdorffLoss, self).__init__()

    def forward(self, set1, set2):
        """
        Compute the Averaged Hausdorff Distance function
         between two unordered sets of points (the function is symmetric).
         Batches are not supported, so squeeze your inputs first!
        :param set1: Tensor where each row is an N-dimensional point.
        :param set2: Tensor where each row is an N-dimensional point.
        :return: The Averaged Hausdorff Distance between set1 and set2.
        """

        assert set1.ndimension() == 2, 'got %s' % set1.ndimension()
        assert set2.ndimension() == 2, 'got %s' % set2.ndimension()

        assert set1.size()[1] == set2.size()[1], \
            'The points in both sets must have the same number of dimensions, got %s and %s.'\
            % (set2.size()[1], set2.size()[1])

        d2_matrix = cdist(set1, set2)

        term_1 = torch.sum(torch.min(d2_matrix, 1)[0])
        term_2 = torch.sum(torch.min(d2_matrix, 0)[0])

        res = (term_1+term_2)*0.5

        return res
