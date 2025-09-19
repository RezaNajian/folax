import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
import numpy as np
from mechanical2d_utilities import *

case_dir = os.path.join(os.path.dirname(__file__),f"./mechanical_2d_error_data/")
for pc in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
    ifol_err_ux = np.loadtxt(case_dir + f"err_ux_PC_{pc}.txt")
    ifol_err_p11 = np.loadtxt(case_dir + f"err_p11_PC_{pc}.txt")


    # Ux
    ifol_err_ux_mae = np.sum(ifol_err_ux,axis=1) / ifol_err_ux.shape[-1]
    ifol_err_ux_max = np.max(ifol_err_ux,axis=1)

    # P11
    ifol_err_p11_mae = np.sum(ifol_err_p11,axis=1) / ifol_err_p11.shape[-1]
    ifol_err_p11_max = np.max(ifol_err_p11,axis=1)


    array = ifol_err_p11_mae
    outlier_limit = 100
    nan_indices = np.argwhere(np.isnan(array))
    outlier_indices = np.argwhere(array > outlier_limit)

    elim_indices = np.union1d(nan_indices.flatten(),outlier_indices.flatten())
    indices = np.setdiff1d(np.arange(array.shape[0]),elim_indices)
    indices = np.arange(array.shape[0])

    # filetered
    # Ux
    ifol_err_ux_mae_cleaned = np.sum(ifol_err_ux[indices,:],axis=1) / ifol_err_ux[indices,:].shape[-1]
    ifol_err_ux_max_cleaned = np.max(ifol_err_ux[indices,:],axis=1)

    # P11
    ifol_err_p11_mae_cleaned = np.sum(ifol_err_p11[indices,:],axis=1) / ifol_err_p11[indices,:].shape[-1]
    ifol_err_p11_max_cleaned = np.max(ifol_err_p11[indices,:],axis=1)

    # print("ifol error ux MAE: ", np.mean(ifol_err_ux_mae_cleaned))
    # print("ifol error ux Max: ", np.mean(ifol_err_ux_max_cleaned))
    # print("ifol error p11 MAE: ", np.mean(ifol_err_p11_mae_cleaned))
    # print("ifol error p11 Max: ", np.mean(ifol_err_p11_max_cleaned))
    # print("final shape: ",ifol_err_p11_mae_cleaned.shape)
    # print("indices to be eliminated: ", elim_indices)
    # print("indices that have kept: ", indices)

    np.savetxt(case_dir + f"err_mae_ux_PC_{pc}_cleaned.txt",ifol_err_ux_mae_cleaned)
    np.savetxt(case_dir + f"err_max_ux_PC_{pc}_cleaned.txt",ifol_err_ux_max_cleaned)
    np.savetxt(case_dir + f"err_mae_p11_PC_{pc}_cleaned.txt",ifol_err_p11_mae_cleaned)
    np.savetxt(case_dir + f"err_max_p11_PC_{pc}_cleaned.txt",ifol_err_p11_max_cleaned)