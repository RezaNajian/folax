import sys
import os
import optax
import numpy as np
from flax import nnx
import jax
from fol.loss_functions.phasefield_2D_fe_quad import AllenCahnLoss2DQuad
from fol.mesh_input_output.mesh import Mesh
from fol.controls.no_control import NoControl
from fol.deep_neural_networks.implicit_transient_parametric_operator_learning import ImplicitParametricOperatorLearning
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from siren_nn import Siren
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# directory & save handling
working_directory_name = 'siren_implicit_AllenCahn_2D'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":100,
                "T_left":1.0,"T_right":-1.0}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

# create fe-based loss function
bc_dict = {"T":{"left":model_settings["T_left"],"right":model_settings["T_right"]}}#

material_dict = {"rho":1.0,"cp":1.0,"dt":0.005,"epsilon":0.1}
thermal_loss_2d = AllenCahnLoss2DQuad("thermal_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "num_gp":2,
                                                                            "material_dict":material_dict},
                                                                            fe_mesh=fe_mesh)

no_control = NoControl("No_Control",fe_mesh)

fe_mesh.Initialize()
thermal_loss_2d.Initialize()
no_control.Initialize()

# create some random coefficients & K for training
create_random_coefficients = True
if create_random_coefficients:
    # number_of_random_samples = 200
    # coeffs_matrix,K_matrix = create_random_fourier_samples(_control,number_of_random_samples)
    # export_dict = model_settings.copy()
    # export_dict["coeffs_matrix"] = coeffs_matrix
    # export_dict["x_freqs"] = fourier_control.x_freqs
    # export_dict["y_freqs"] = fourier_control.y_freqs
    # export_dict["z_freqs"] = fourier_control.z_freqs
    # with open(f'fourier_control_dict_N_{model_settings["N"]}.pkl', 'wb') as f:
    #     pickle.dump(export_dict,f)
    # def generate_random_pattern(rows, cols, mean=0, std=1):
    #     """Generate a random pattern in a 2-D domain using Gaussian random process."""
    #     # Generate a 2-D array with random values from a Gaussian distribution
    #     random_pattern = np.random.normal(loc=mean, scale=std, size=(rows, cols))
    #     random_pattern = (random_pattern-np.min(random_pattern))/(np.max(random_pattern)-np.min(random_pattern))
    #     return random_pattern
    # coeffs_matrix = generate_random_pattern(model_settings["N"], model_settings["N"], mean=0, std=1)
    def generate_random_smooth_pattern(L,N):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        X1, X2 = np.meshgrid(x, y)
        X = np.vstack([X1.ravel(), X2.ravel()]).T
        u = gp.sample_y(X, n_samples=1, random_state=0).ravel()
        scaled_u = 2.0*(u - np.min(u)) / (np.max(u) - np.min(u)) -1.0
        return scaled_u.reshape(1,-1)
    
    def generate_double_bubble(L,N,epsilon):
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        u = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                func1 = np.tanh((0.2-np.sqrt((x[i]-0.6*L)**2 + (y[j]-0.5*L)**2))/epsilon)
                func2 = np.tanh((0.2-np.sqrt((x[i]-0.4*L)**2 + (y[j]-0.5*L)**2))/epsilon)
                if func1>func2:
                    u[i,j] = func1
                else:
                    u[i,j] = func2
        return u.reshape(1,-1)
    coeffs_matrix = generate_double_bubble(model_settings["L"],model_settings["N"],material_dict["epsilon"])

else:
    pass
    # with open(f'fourier_control_dict_N_{model_settings["N"]}.pkl', 'rb') as f:
    #     loaded_dict = pickle.load(f) 
    # coeffs_matrix = loaded_dict["coeffs_matrix"]
    # coeffs_matrix = jnp.load("gaussian_kernel_50000_N21.npy")


# K_matrix = []
T_matrix = no_control.ComputeBatchControlledVariables(coeffs_matrix)

export_Ks = False
# if export_Ks:
#     for i in range(K_matrix.shape[0]):
#         fe_mesh[f'K_{i}'] = np.array(K_matrix[i,:])
#     fe_mesh.Finalize(export_dir=case_dir)
#     exit()

# specify id of the K of interest
eval_id = 0

# design siren NN for learning
siren_NN = Siren(4,1,[50,50])

# design NN for learning
# class MLP(nnx.Module):
#     def __init__(self, in_features: int, dmid: int, out_features: int, *, rngs: nnx.Rngs):
#         self.dense1 = nnx.Linear(in_features, dmid, rngs=rngs,kernel_init=nnx.initializers.zeros,bias_init=nnx.initializers.zeros)
#         self.dense2 = nnx.Linear(dmid, dmid, rngs=rngs,kernel_init=nnx.initializers.zeros,bias_init=nnx.initializers.zeros)
#         self.dense3 = nnx.Linear(dmid, out_features, rngs=rngs,kernel_init=nnx.initializers.zeros,bias_init=nnx.initializers.zeros)
#         self.in_features = in_features
#         self.out_features = out_features

#     def __call__(self, x: jax.Array) -> jax.Array:
#         x = self.dense1(x)
#         x = jax.nn.swish(x)
#         x = self.dense2(x)
#         x = jax.nn.swish(x)
#         x = self.dense3(x)
#         return x
    
# fol_net = MLP(4,
#               100,
#               len(thermal_loss_2d.dofs),
#               rngs=nnx.Rngs(0))
lr = 1e-3
# create fol optax-based optimizer
chained_transform = optax.chain(optax.normalize_by_update_norm(),
                                optax.adam(lr))

# create fol
fol = ImplicitParametricOperatorLearning(name="dis_fol",control=no_control,
                                        loss_function=thermal_loss_2d,
                                        flax_neural_network=siren_NN,
                                        optax_optimizer=chained_transform,
                                        checkpoint_settings={"restore_state":False,
                                        "state_directory":case_dir+"/flax_state"},
                                        working_directory=case_dir)

fol.Initialize()

t_init = 0.0
t_current = t_init
num_steps = 5
FOL_T_temp = coeffs_matrix.flatten()
FOL_T = np.zeros((fe_mesh.GetNumberOfNodes(),num_steps))
# For the first time step
fol.Train(train_set=(jnp.concatenate((jnp.array([t_current]),FOL_T_temp)).reshape(-1,1).T,),batch_size=100,
            convergence_settings={"num_epochs":2000,"relative_error":1e-100},
            plot_settings={"plot_save_rate":1000},
            save_settings={"save_nn_model":True})
FOL_T_temp = np.array(fol.Predict(jnp.array([t_current]))).reshape(-1)
FOL_T[:,0] = FOL_T_temp
# For the subsequent time steps the checkpoint function should be activated
for i in range(num_steps-1):
    t_current += material_dict["dt"]
    chained_transform = optax.chain(optax.normalize_by_update_norm(),
                                optax.adam(lr))
    fol = ImplicitParametricOperatorLearning(name="dis_fol",control=no_control,
                                            loss_function=thermal_loss_2d,
                                            flax_neural_network=siren_NN,
                                            optax_optimizer=chained_transform,
                                            checkpoint_settings={"restore_state":True,
                                            "state_directory":case_dir+"/flax_state"},
                                            working_directory=case_dir)
    fol.Initialize()
    fol.Train(train_set=(jnp.concatenate((jnp.array([t_current]),FOL_T_temp)).reshape(-1,1).T,),batch_size=100,
                convergence_settings={"num_epochs":2000,"relative_error":1e-100},
                plot_settings={"plot_save_rate":1000},
                save_settings={"save_nn_model":True})
    FOL_T_temp = np.array(fol.Predict(jnp.array([t_current]))).reshape(-1)
    FOL_T[:,i+1] = FOL_T_temp

fe_mesh['T_FOL'] = FOL_T#.reshape((fe_mesh.GetNumberOfNodes(), 1))
# # solve FE here
# fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-6,"atol":1e-6,
#                                             "maxiter":1000,"pre-conditioner":"ilu"},
#                 "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
#                                             "maxiter":10,"load_incr":5}}
# linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver",thermal_loss_2d,fe_setting)
# linear_fe_solver.Initialize()
# FE_T = np.array(linear_fe_solver.Solve(T_matrix[eval_id],np.zeros(fe_mesh.GetNumberOfNodes())))  
# fe_mesh['T_FE'] = FE_T.reshape((fe_mesh.GetNumberOfNodes(), 1))

# absolute_error = abs(FOL_T.reshape(-1,1)- FE_T.reshape(-1,1))
# fe_mesh['abs_error'] = absolute_error.reshape((fe_mesh.GetNumberOfNodes(), 1))

plot_mesh_vec_data_phasefield(1,[coeffs_matrix[eval_id],FOL_T[:,0], FOL_T[:,2], FOL_T[:,-1]],#,absolute_error
                   ["T_init","T_1","T_3","T_fin"],
                   fig_title="Initial condition and implicit FOL solution",cmap = "jet",
                   file_name=os.path.join(case_dir,"FOL-T-dist.png"))
# plot_mesh_vec_data(1,[T_matrix[eval_id,:],FE_T],
#                    ["T_init","T"],
#                    fig_title="conductivity and FEM solution",cmap = "viridis",
#                    file_name=os.path.join(case_dir,"FEM-KT-dist.png"))

fe_mesh.Finalize(export_dir=case_dir)
