import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

import numpy as np
import pickle
import matplotlib.pyplot as plt
from mechanical2d_utilities import *


### Script's goal:
####### the following script is to 
####### Calculate error analysis on the data



# directory & save handling
working_directory_name = "mechanical_2d_base_from_ifol_meta"
case_dir = os.path.join('.', working_directory_name)

bc = 0.5    # right boundary condition
ph = 0.1   # phase contrast

# load data dir
data_directory_name = f"{working_directory_name}_data"
data_dir = os.path.join('.', data_directory_name)


# load data into a list of tuples corresponding to resolutions 21, 41, and 81. 
# i.e.  U = [(U_ifol, U_HFE)]
#       P = [(P_ifol, P_HFE)]
#       err_U = [err_U_ifol]
#       err_P = [err_P_ifol]
U_dict_to_check = {}
file_path = os.path.join(data_dir,f"U_base_res_81_bc_{bc}_phase_contrast_{ph}.pkl")
with open(file_path, 'rb') as f:
    U_dict_to_check = pickle.load(f)

result_dict = {}
result_dict["UV_HFE"] = {}
result_dict["UV_iFOL"] = {}
result_dict["P_HFE"] = {}
result_dict["P_iFOL"] = {}
result_dict["err_UV"] = {}
result_dict["err_P"] = {}

res_list = [21,41,81]
numberof_sample = []
index_list = [[],[],[]]
for indx,res in enumerate(res_list):
    num_sample = 0
    U_dict = {}
    UV_HFE, UV_iFOL = [], []
    P_HFE, P_iFOL = [], []
    err_UV = []
    err_P = []
    file_path = os.path.join(data_dir,f"U_base_res_{res}_bc_{bc}_phase_contrast_{ph}.pkl")
    with open(file_path, 'rb') as f:
        U_dict = pickle.load(f)
    
    # for eval_id in range(K_matrix_2.shape[0]):
    for eval_id in range(400):
        if U_dict_to_check.get(f"U_FE_81_{eval_id}") is not None: 
            if U_dict[f"U_FE_{res}_{eval_id}"] is not None:

                UV_HFE.append(U_dict[f'U_FE_{res}_{eval_id}'].flatten())
                UV_iFOL.append(U_dict[f'U_iFOL_{res}_{eval_id}'].flatten())
                P_HFE.append(U_dict[f'Stress_FE_{res}_{eval_id}'].flatten())
                P_iFOL.append(U_dict[f'Stress_iFOL_{res}_{eval_id}'].flatten())
                err_UV.append(U_dict[f'abs_error_{res}_{eval_id}'].flatten())
                err_P.append(U_dict[f'abs_stress_error_{res}_{eval_id}'].flatten())
                
                num_sample += 1
                index_list[indx].append(eval_id)

    # numberof_sample.append(num_sample)
    numberof_sample = num_sample

    # save to dictionary as np.arrays
    result_dict["UV_HFE"][f"{res}"] = np.array(UV_HFE)
    result_dict["UV_iFOL"][f"{res}"] = np.array(UV_iFOL)
    result_dict["P_HFE"][f"{res}"] = np.array(P_HFE)
    result_dict["P_iFOL"][f"{res}"] = np.array(P_iFOL)
    result_dict["err_UV"][f"{res}"] = np.array(err_UV)
    result_dict["err_P"][f"{res}"] = np.array(err_P)

# error analysis
err_dict = {}
err_dict["UX_MAE"] = {}
err_dict["UY_MAE"] = {}
err_dict["P11_MAE"] = {}
err_dict["UX_MAX-AE"] = {}
err_dict["UY_MAX-AE"] = {}
err_dict["P11_MAX-AE"] = {}

print("number of sample: ",numberof_sample)
sample_num = min(numberof_sample,200)
idx = np.ix_(range(sample_num))
idx = np.ix_([i for i in range(sample_num) if i not in (14, 21)])   # defected samples

# idx = np.ix_(np.array(index_list[-1]))
index_to_check = 21
for res in res_list:
    # for Ux
    # mean absolute errors
    HFEM_UV,iFOL_UV = result_dict["UV_HFE"][f"{res}"][idx],result_dict["UV_iFOL"][f"{res}"][idx]
    HFEM_U,iFOL_U = HFEM_UV[:,::2],iFOL_UV[:,::2]
    print("shape of HFEM: ",HFEM_U.shape)
    print("check for nan for HFEM U: ", np.argwhere(np.isnan(HFEM_U)))
    plt.subplot(1,2,1)
    plt.imshow(HFEM_U[index_to_check,:].reshape(res,res), cmap="coolwarm")
    plt.subplot(1,2,2)
    plt.imshow(iFOL_U[index_to_check,:].reshape(res,res), cmap="coolwarm")
    plt.show()
    absolute_error_test_ux = np.abs(iFOL_U- HFEM_U)
    test_mae_err_for_samples_ux = np.sum(absolute_error_test_ux,axis=1) / absolute_error_test_ux.shape[-1]
    test_mae_err_total_ux = np.mean(test_mae_err_for_samples_ux)
    err_dict["UX_MAE"][f"{res}"] = float(test_mae_err_total_ux)
    print(f"mean absolute error of Ux for res {res}: ",test_mae_err_total_ux)

    # max absolute errors
    test_max_err_for_samples_ux = np.max(absolute_error_test_ux,axis=1)
    test_max_err_total_ux = np.mean(test_max_err_for_samples_ux)
    err_dict["UX_MAX-AE"][f"{res}"] = float(test_max_err_total_ux)
    print(f"max absolute error of Ux for res {res}: ",test_max_err_total_ux)



    # for Uy
    # mean absolute errors
    HFEM_UV,iFOL_UV = result_dict["UV_HFE"][f"{res}"][idx],result_dict["UV_iFOL"][f"{res}"][idx]
    HFEM_V,iFOL_V = HFEM_UV[:,1::2],iFOL_UV[:,1::2]
    print("shape of HFEM V: ",HFEM_V.shape)
    print("check for nan for HFEM V: ", np.argwhere(np.isnan(HFEM_V)))
    absolute_error_test_uy = np.abs(iFOL_V- HFEM_V)
    test_mae_err_for_samples_uy = np.sum(absolute_error_test_uy,axis=1) / absolute_error_test_uy.shape[-1]
    test_mae_err_total_uy = np.mean(test_mae_err_for_samples_uy)
    err_dict["UY_MAE"][f"{res}"] = float(test_mae_err_total_uy)
    print(f"mean absolute error of Uy for res {res}: ",test_mae_err_total_uy)

    # max absolute errors
    test_max_err_for_samples_uy = np.max(absolute_error_test_uy,axis=1)
    test_max_err_total_uy = np.mean(test_max_err_for_samples_uy)
    err_dict["UY_MAX-AE"][f"{res}"] = float(test_max_err_total_uy)
    print(f"max absolute error of Uy for res {res}: ",test_max_err_total_uy)



    # for P11
    # mean absolute errors
    HFEM_P,iFOL_P = result_dict["P_HFE"][f"{res}"][idx],result_dict["P_iFOL"][f"{res}"][idx]
    HFEM_P11,iFOL_P11 = HFEM_P[:,::3],iFOL_P[:,::3]
    print("shape of HFEM P11: ",HFEM_P11.shape)
    print("check for nan for P11: ", np.argwhere(np.isnan(HFEM_P11)))
    
    plt.subplot(1,2,1)
    plt.imshow(HFEM_P11[index_to_check,:].reshape(res,res), cmap="coolwarm")
    plt.subplot(1,2,2)
    plt.imshow(iFOL_P11[index_to_check,:].reshape(res,res), cmap="coolwarm")
    plt.show()
    absolute_error_test_p11 = np.abs(iFOL_P11- HFEM_P11)
    print("for large values: ", np.argwhere(absolute_error_test_p11 > 10))
    test_mae_err_for_samples_p11 = np.sum(absolute_error_test_p11,axis=1) / absolute_error_test_p11.shape[-1]
    test_mae_err_total_p11 = np.mean(test_mae_err_for_samples_p11)
    err_dict["P11_MAE"][f"{res}"] = float(test_mae_err_total_p11)
    print(f"mean absolute error of P11 for res {res}: ",test_mae_err_total_p11)

    # max absolute errors
    test_max_err_for_samples_p11 = np.max(absolute_error_test_p11,axis=1)
    test_max_err_total_p11 = np.mean(test_max_err_for_samples_p11)
    err_dict["P11_MAX-AE"][f"{res}"] = float(test_max_err_total_p11)
    print(f"max absolute error of P11 for res {res}: ",test_max_err_total_p11)


output_directory_name = f"{working_directory_name}_postprocessing"
output_dir = os.path.join('.',output_directory_name)
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir,f'output_{ph}_json')
data_to_dump = {"microstructure data": {"phase contrast": ph,
                                        "Boundary Condition Ux right":bc,
                                        "number of microstructures": sample_num,
                                        "type of microstructures": ["fourier", "voronoi dual", "voronoi multi", "tpms"]},
                "Error Analysis Results":err_dict}
with open(output_filename, 'w') as f:
    json.dump(data_to_dump, f, indent=4)







                    
                    
    