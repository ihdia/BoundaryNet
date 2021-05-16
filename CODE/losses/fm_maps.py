import numpy as np
import skfmm

# ----------- Distance Maps formation -----------
def compute_edts_forPenalizedLoss(GT):

    GT = np.array(GT,dtype = "uint8")
    GT = np.reshape(GT,(GT.shape[0],GT.shape[1],1))
    GT = np.transpose(GT, (2, 0, 1))
    res = np.zeros(GT.shape)
    temp = GT[0]
    for i in range(GT.shape[0]):
        posmask = GT[i]
        negmask = posmask
        posmask = (posmask*-2)+1
        pos_edt = skfmm.distance(posmask, dx=1e-3)
        bias1 = np.mean(pos_edt)

        pos_edt = (np.max(pos_edt)-pos_edt)
        pos_edt = np.abs(pos_edt)
        res[i] = pos_edt/np.max(pos_edt)
    ress = res+1
    return ress