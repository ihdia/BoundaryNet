import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_cdt
import torch
from skimage.io import imsave

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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