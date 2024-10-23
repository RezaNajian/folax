import sys
import os
import optax
from flax import nnx
import jax
import numpy as np
from fol.loss_functions.mechanical_2D_fe_quad import MechanicalLoss2D
from fol.mesh_input_output.mesh import Mesh
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.implicit_parametric_operator_learning import ImplicitParametricOperatorLearning
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
import pickle, time

# directory & save handling
working_directory_name = 'implicit_mechanical_2D'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":51,
                "Ux_left":0.0,"Ux_right":0.05,
                "Uy_left":0.0,"Uy_right":0.05}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

# create fe-based loss function
bc_dict = {"Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
           "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]}}

material_dict = {"young_modulus":1,"poisson_ratio":0.3}
mechanical_loss_2d = MechanicalLoss2D("mechanical_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "num_gp":2,
                                                                            "material_dict":material_dict},
                                                                            fe_mesh=fe_mesh)

fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                            "beta":20,"min":1e-1,"max":1}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)


fe_mesh.Initialize()
mechanical_loss_2d.Initialize()
fourier_control.Initialize()

# create some random coefficients & K for training
create_random_coefficients = False
if create_random_coefficients:
    number_of_random_samples = 200
    coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
    export_dict = model_settings.copy()
    export_dict["coeffs_matrix"] = coeffs_matrix
    export_dict["x_freqs"] = fourier_control.x_freqs
    export_dict["y_freqs"] = fourier_control.y_freqs
    export_dict["z_freqs"] = fourier_control.z_freqs
    with open(f'fourier_control_dict_N_{model_settings["N"]}.pkl', 'wb') as f:
        pickle.dump(export_dict,f)
else:
    with open(f'fourier_control_dict_N_{model_settings["N"]}.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    coeffs_matrix = loaded_dict["coeffs_matrix"]

K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

# specify id of the K of interest
eval_id = 25

# design NN for learning
class MLP(nnx.Module):
    def __init__(self, in_features: int, dmid: int, out_features: int, *, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(in_features, dmid, rngs=rngs)
        self.dense2 = nnx.Linear(dmid, dmid, rngs=rngs)
        self.dense3 = nnx.Linear(dmid, out_features, rngs=rngs)
        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.dense1(x)
        x = jax.nn.swish(x)
        x = self.dense2(x)
        x = jax.nn.swish(x)
        x = self.dense3(x)
        return x

# here we create netwrok for implicit learning
# Note : 
#       input size = 3 for (x,y,z) + number of control variables
#       output size = number of degree of freedoms 
fol_net = MLP(3+fourier_control.GetNumberOfVariables(),
              100,
              len(mechanical_loss_2d.dofs),
              rngs=nnx.Rngs(0))

# create fol optax-based optimizer
chained_transform = optax.chain(optax.adam(1e-3))

# create fol
fol = ImplicitParametricOperatorLearning(name="dis_fol",control=fourier_control,
                                        loss_function=mechanical_loss_2d,
                                        flax_neural_network=fol_net,
                                        optax_optimizer=chained_transform,
                                        checkpoint_settings={"restore_state":False,
                                        "state_directory":case_dir+"/flax_state"},
                                        working_directory=case_dir)

fol.Initialize()

# here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
fol.Train(train_set=(coeffs_matrix[eval_id,:].reshape(-1,1).T,),batch_size=100,
            convergence_settings={"num_epochs":500,"relative_error":1e-100},
            plot_settings={"plot_save_rate":1000})

FOL_UV = np.array(fol.Predict(coeffs_matrix[eval_id,:].reshape(-1,1).T)).reshape(-1)
fe_mesh['U_FOL'] = FOL_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))

# solve FE here
fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-6,"atol":1e-6,
                                            "maxiter":1000,"pre-conditioner":"ilu"},
                "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                            "maxiter":10,"load_incr":5}}
linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver",mechanical_loss_2d,fe_setting)
linear_fe_solver.Initialize()
FE_UV = np.array(linear_fe_solver.Solve(K_matrix[eval_id],np.zeros(2*fe_mesh.GetNumberOfNodes())))  
fe_mesh['U_FE'] = FE_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))

absolute_error = abs(FOL_UV.reshape(-1,1)- FE_UV.reshape(-1,1))
fe_mesh['abs_error'] = absolute_error.reshape((fe_mesh.GetNumberOfNodes(), 2))


plot_mesh_vec_data(1,[FOL_UV[0::2],FOL_UV[1::2],absolute_error[0::2],absolute_error[1::2]],
                   ["U","V","abs_error_U","abs_error_V"],
                   fig_title="implicit FOL solution and error",
                   file_name=os.path.join(case_dir,"FOL-UV-dist.png"))
plot_mesh_vec_data(1,[K_matrix[eval_id,:],FE_UV[0::2],FE_UV[1::2]],
                   ["K","U","V"],
                   fig_title="conductivity and FEM solution",
                   file_name=os.path.join(case_dir,"FEM-KUV-dist.png"))

fe_mesh.Finalize(export_dir=case_dir)
