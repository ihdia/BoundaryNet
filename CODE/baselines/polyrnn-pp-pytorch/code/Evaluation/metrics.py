import torch
import torch.nn.functional as F
import numpy as np
from Utils import utils
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,generate_binary_structure

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # test for emptiness
    if 0 == np.count_nonzero(result): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference): 
        raise RuntimeError('The second supplied array does not contain any binary object.')    
            
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    # print(result_border+0)
    # result_border = (result_border+0).astype(np.float32)
    # imsave("./test_comp1/" +"truth.jpg", result_border)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    # print(reference_border)
    # reference_border = (reference_border + 0).astype(np.float32)
    # imsave("./test_comp1/" +"truth.jpg", reference_border)
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    # print(dt)
    reference_border = (reference_border + 0).astype(np.float32)
    # imsave("./test_comp1/" +"truth.jpg", reference_border)
    sds = dt[result_border]
    
    return sds


def hd(result, reference, voxelspacing=None, connectivity=1):
    """
    Hausdorff Distance.
    
    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. It is defined as the maximum surface distance between the objects.
    
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
        
    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of 
        elements along each dimension, which is usually given in mm.
        
    See also
    --------
    :func:`assd`
    :func:`asd`
    
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd

def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.

    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of 
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`hd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95

def accuracy(poly, mask, pred_polys, grid_size):
    """
    Computes prediction accuracy

    poly: [batch_size, time_steps]
    pred_polys: [batch_size, time_steps,]
    Each element stores y*grid_size + x, or grid_size**2 for EOS

    mask: [batch_size, time_steps,]
    The mask of valid time steps in the GT poly. It is manipulated
    inside this function!

    grid_size: size of the grid in which the polygons are in    
    """
    idxs = np.argmax(pred_polys, axis=-1)
    for i,idx in enumerate(idxs):
        if pred_polys[i,idx] == grid_size**2:
            # If EOS
            if idx > np.sum(mask[i,:]):
                # If there are more predictions than
                # ground truth points, then extend mask
                mask[i, :idx] = 1.

        else:
            # If no EOS was predicted
            mask[i, :] = 1.
    
    corrects = pred_polys == poly

    corrects = corrects * mask
    percentage = np.sum(corrects, axis=-1)*1.0/np.sum(mask, axis=-1)

    return np.mean(percentage)

def train_accuracy(poly, mask, pred_polys, grid_size):
    """
    Computes prediction accuracy with GT masks

    poly: [batch_size, time_steps]
    pred_polys: [batch_size, time_steps,]
    Each element stores y*grid_size + x, or grid_size**2 for EOS

    mask: [batch_size, time_steps,]

    grid_size: size of the grid in which the polygons are in    
    accepts grid_size to be compatible with accuracy()
    """
    corrects = (pred_polys == poly).astype(np.float32)

    corrects = corrects * mask

    percentage = np.sum(corrects, axis=-1)*1.0/np.sum(mask, axis=-1)

    return np.mean(percentage)

def iou_from_mask(pred, gt):
    """
    Compute intersection over the union.
    Args:
        pred: Predicted mask
        gt: Ground truth mask
    """
    pred = pred.astype(np.bool)
    gt = gt.astype(np.bool)

    true_negatives = np.count_nonzero(np.logical_and(np.logical_not(gt), np.logical_not(pred)))
    false_negatives = np.count_nonzero(np.logical_and(gt, np.logical_not(pred)))
    false_positives = np.count_nonzero(np.logical_and(np.logical_not(gt), pred))
    true_positives = np.count_nonzero(np.logical_and(gt, pred))

    union = float(true_positives + false_negatives)
    intersection = float(true_positives)

    iou = intersection / union if union > 0. else 0.

    return iou

def iou_from_poly(pred, gt, width, height):
    """
    Compute IoU from poly. The polygons should
    already be in the final output size

    pred: list of np arrays of predicted polygons
    gt: list of np arrays of gt polygons
    grid_size: grid_size that the polygons are in

    """
    masks = np.zeros((2, height, width), dtype=np.uint8)

    pred[:,0] = pred[:,0] / float(28) * width
    pred[:,1] = pred[:,1] / float(28) * height

    gt[:,0] = gt[:,0] / float(28) * width
    gt[:,1] = gt[:,1] / float(28) * height

    if not isinstance(pred, list):
        pred = [pred]
    if not isinstance(gt, list):
        gt = [gt]

    for p in pred: 
        masks[0] = utils.draw_poly(masks[0], p)

    for g in gt:
        masks[1] = utils.draw_poly(masks[1], g)

    return iou_from_mask(masks[0], masks[1]), masks

def hd_from_poly(pred, gt, width, height):
    """
    Compute IoU from poly. The polygons should
    already be in the final output size

    pred: list of np arrays of predicted polygons
    gt: list of np arrays of gt polygons
    grid_size: grid_size that the polygons are in

    """
    masks = np.zeros((2, height, width), dtype=np.uint8)

    # pred[:,0] = pred[:,0] / float(28) * width
    # pred[:,1] = pred[:,1] / float(28) * height

    # gt[:,0] = gt[:,0] / float(28) * width
    # gt[:,1] = gt[:,1] / float(28) * height

    if not isinstance(pred, list):
        pred = [pred]
    if not isinstance(gt, list):
        gt = [gt]

    for p in pred: 
        masks[0] = utils.draw_poly2(masks[0], p)

    for g in gt:
        masks[1] = utils.draw_poly2(masks[1], g)
    # try:
    hd77 = hd(masks[0], masks[1])
    hd779 = hd95(masks[0], masks[1])
    # except:
    #     hd77 = 0
    #     hd779 = 0
    return hd77, hd779
