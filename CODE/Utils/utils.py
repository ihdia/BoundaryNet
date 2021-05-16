import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_cdt
import torch
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,generate_binary_structure
from skimage.io import imsave

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ----------- IoU and Accuracy -----------
def compute_iou_and_accuracy(arrs, edge_mask1):
    intersection = cv2.bitwise_and(arrs, edge_mask1)
    union = cv2.bitwise_or(arrs, edge_mask1)

    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)

    iou = (intersection_sum) / (union_sum)

    total = np.sum(arrs)
    correct_predictions = intersection_sum

    accuracy = correct_predictions / total
    # print(iou, accuracy)

    return iou, accuracy

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

    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    # print(dt)
    reference_border = (reference_border + 0).astype(np.float32)

    sds = dt[result_border]
    
    return sds

# ----------- Hausdorff Distance metrics -----------

def hd(result, reference, voxelspacing=None, connectivity=1):
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd

def hd95(result, reference, voxelspacing=None, connectivity=1):
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95


def uniformsample_batch(batch, num):
    final = []
    for i in batch:
        i1 = np.asarray(i).astype(int)
        a = uniformsample(i1, num)
        a = torch.from_numpy(a)
        a = a.long()
        final.append(a)
    return final

def uniformsample(pgtnp_px2, newpnum):
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > newpnum:
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum
        return pgtnp_kx2
    # we need to add gt points
    # we simply add it uniformly
    else:
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        if edgenumsum != newpnum:

            if edgenumsum > newpnum:

                id = -1
                passnum = edgenumsum - newpnum
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        assert np.sum(edgenum) == newpnum

        psample = []
        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i];

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp



def get_edge_mask(poly, mask):
    """
    Generate edge mask
    """
    h = mask.shape[0]
    w = mask.shape[1]
    gt_poly = np.zeros((poly.shape[0],poly.shape[1]),np.int32)
    gt_poly[:,0] = np.floor(poly[:,0]*w)
    gt_poly[:,1] = np.floor(poly[:,1]*h)
    # print(gt_poly[:,0], gt_poly[:,1])    
    cv2.polylines(mask, np.int32([gt_poly]),True,[1], thickness = 1)
    # cv2.fillPoly(mask, np.int32([gt_poly]),[255])
    # imsave("./test33/"+str(poly.shape[0])+"edge.jpg",mask[0])
    # imsave("./test33/"+str(poly.shape[0])+"edgegt.jpg",mask[1])

    return mask

def get_poly_mask(poly, mask):
    """
    Generate edge mask
    """
    h = mask.shape[0]
    w = mask.shape[1]
    gt_poly = np.zeros((poly.shape[0],poly.shape[1]),np.int32)
    gt_poly[:,0] = np.floor(poly[:,0]*w)
    gt_poly[:,1] = np.floor(poly[:,1]*h)
        # print(gt_poly[:,0], gt_poly[:,1])    
    # cv2.polylines(mask, np.int32([gt_poly]),True,[1], thickness = 1)
    cv2.fillPoly(mask, np.int32([gt_poly]),[1])
    # print(np.unique(mask[0]))
    # cv2.imwrite("gtmask"+str(step)+'.png',mask[0])



    # imsave("./test33/"+str(poly.shape[0])+"edge.jpg",mask[0])
    # imsave("./test33/"+str(poly.shape[0])+"edgegt.jpg",mask[1])

    return mask

def get_original_mask(poly, mask):
    """
    Generate edge mask
    """
    h = mask.shape[0]
    w = mask.shape[1]
    gt_poly = np.zeros((poly.shape[0],poly.shape[1]),np.int32)
    gt_poly[:,0] = np.floor(poly[:,0]*w)
    gt_poly[:,1] = np.floor(poly[:,1]*h)
    # print(gt_poly[:,0], gt_poly[:,1])    
    # cv2.polylines(mask, np.int32([gt_poly]),True,[1], thickness = 1)
    cv2.fillPoly(mask, np.int32([gt_poly]),[1])
    # imsave("./test33/"+str(poly.shape[0])+"edge.jpg",mask[0])
    # imsave("./test33/"+str(poly.shape[0])+"edgegt.jpg",mask[1])

    return mask

def get_fp_mask(poly,mask):
    h = mask.shape[0]
    w = mask.shape[1]
    x = np.int32(np.floor(poly[0,0]*w))
    y = np.int32(np.floor(poly[0,1]*h))
    mask[y,x] = 1.0
    # if(y<=14 and x<=190 and x>=1 and y>=1):
    #     mask[y,x+1] = 1.0
    #     mask[y,x-1] = 1.0
    #     mask[y+1,x] = 1.0
    #     mask[y-1,x] = 1.0
    return mask

def get_vertices_mask(poly, mask):
    """
    Generate a vertex mask
    """
    h = mask.shape[0]
    w = mask.shape[1]
    gt_poly = np.zeros((poly.shape[0],poly.shape[1]),np.int32)
    gt_poly[:,0] = np.floor(poly[:,0]*w)
    gt_poly[:,1] = np.floor(poly[:,1]*h)

    mask[gt_poly[:, 1], gt_poly[:, 0]] = 1.0

    return mask

def get_previous_mask(poly,mask,t):
    mask = torch.zeros(1, 1, 25, 60, device=device)
    h = 25
    w = 60
    x = np.int32(np.floor(poly[0,t,0]*w))
    y = np.int32(np.floor(poly[0,t,1]*h))
    mask[0,0,y,x] = 1
    # if(y<=14 and x<=190 and x>=1 and y>=1):
    #     mask[0,0,y,x+1] = 1.0
    #     mask[0,0,y,x-1] = 1.0
    #     mask[0,0,y+1,x] = 1.0
    #     mask[0,0,y-1,x] = 1.0
    return mask


def get_instance_mask(poly,mask):
    h = 25
    w = 60
    masks = []
    for tr in range(poly.shape[0]):
        # print(poly[tr,0],poly[tr,1])
        x = np.int32(np.floor(poly[tr,0]*w))
        y = np.int32(np.floor(poly[tr,1]*h))
        # print(y,x)
        mask[y,x] = 1.0
        # if(y<=14 and x<=190 and x>=1 and y>=1):
        #     mask[y,x+1] = 1.0
        #     mask[y,x-1] = 1.0
        #     mask[y+1,x] = 1.0
        #     mask[y-1,x] = 1.0
        mask1 = mask.flatten()
        if(tr == poly.shape[0]-1):
            mask1 = np.append(mask1,[1.0])
        else:
            mask1 = np.append(mask1,[0.0])
        masks.append(mask1)
        # print(y,x)
        mask[y,x] = 0.0
        # if(y<=14 and x<=190 and x>=1 and y>=1):
        #     mask[y+1,x] = 0.0
        #     mask[y-1,x] = 0.0
        #     mask[y,x+1] = 0.0
        #     mask[y,x-1] = 0.0
    return np.asarray(masks, dtype=np.float32)



def class_to_grid(poly, out_tensor):
    """
    NOTE: Torch function
    accepts out_tensor to do it inplace

    poly: [batch, ]
    out_tensor: [batch, 1, grid_size, grid_size]
    """
    out_tensor.zero_()
    # Remove old state of out_tensor

    b = 0
    for i in poly:
        if i < 16 * 192:
            x = (i%192).long()
            y = (i/16).long()
            out_tensor[0,0,y,x] = 1
        b += 1

    return out_tensor



def dt_targets_from_class(poly):
    """
    NOTE: numpy function!
    poly: [bs, time_steps], each value in [0, grid*size**2+1)
    grid_size: size of the grid the polygon is in
    dt_threshold: threshold for smoothing in dt targets

    returns: 
    full_targets: [bs, time_steps, grid_size**2+1] array containing 
    dt smoothed targets to be used for the polygon loss function
    """
    full_targets = []
    for b in range(poly.shape[0]):
        targets = []
        for p in poly[b]:
            t = np.zeros(16*192+1, dtype=np.int32)
            t[p] += 1

            if p != 16*192:#EOS
                spatial_part = t[:-1]
                spatial_part = np.reshape(spatial_part, [16, 192, 1])

                # Invert image
                spatial_part = -1 * (spatial_part - 1)
                # Compute distance transform
                spatial_part = distance_transform_cdt(spatial_part, metric='taxicab').astype(np.float32)
                # Threshold
                spatial_part = np.clip(spatial_part, 0, dt_threshold)
                # Normalize
                spatial_part /= dt_threshold
                # Invert back
                spatial_part = -1. * (spatial_part - 1.)

                spatial_part /= np.sum(spatial_part)
                spatial_part = spatial_part.flatten()

                t = np.concatenate([spatial_part, [0.]], axis=-1)

            targets.append(t.astype(np.float32))
        full_targets.append(targets)

    return np.array(full_targets, dtype=np.float32)

# def class_to_grid(poly, out_tensor):
#     """
#     NOTE: Torch function
#     accepts out_tensor to do it inplace

#     poly: [batch, ]
#     out_tensor: [batch, 1, grid_size, grid_size]
#     """
#     out_tensor.zero_()
#     # Remove old state of out_tensor

#     b = 0
#     for i in poly:
#         if i < 16 * 192:
#             x = (i%192).long()
#             y = (i/16).long()
#             out_tensor[b,0,y,x] = 1
#         b += 1

#     return out_tensor