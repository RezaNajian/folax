import sys
import os

import numpy as np
from fol.loss_functions.mechanical import MechanicalLoss3DTetra
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.solvers.adjoint_fe_solver import AdjointFiniteElementSolver
from fol.mesh_input_output.mesh import Mesh
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.explicit_parametric_operator_learning import ExplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
from fol.responses.fe_response import FiniteElementResponse
import pickle
import optax
from flax import nnx
import jax
# directory & save handling
working_directory_name = "box_3D_tetra"
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# create mesh_io
fe_mesh = Mesh("fol_io","box_3D_coarse.med",'../meshes/')

# create fe-based loss function
bc_dict = {"Ux":{"left":0.0},
            "Uy":{"left":0.0,"right":-0.05},
            "Uz":{"left":0.0,"right":-0.05}}
material_dict = {"young_modulus":1,"poisson_ratio":0.3}
mechanical_loss_3d = MechanicalLoss3DTetra("mechanical_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                "material_dict":material_dict},
                                                                                fe_mesh=fe_mesh)

# fourier control
fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([2,4,6]),
                            "beta":20,"min":1e-1,"max":1}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)

myresponse = FiniteElementResponse("my_response",response_formula="U.dot(U)",fe_loss=mechanical_loss_3d)

fe_mesh.Initialize()
mechanical_loss_3d.Initialize()
fourier_control.Initialize()
myresponse.Initialize()

with open(f'fourier_control_dict.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

coeffs_matrix = loaded_dict["coeffs_matrix"]

K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

#specify the control 
eval_id = 1

fe_mesh['E'] = K_matrix[eval_id].reshape((fe_mesh.GetNumberOfNodes(), 1))


fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl"}}
first_fe_solver = FiniteElementLinearResidualBasedSolver("first_fe_solver",mechanical_loss_3d,fe_setting)
first_fe_solver.Initialize()
FE_UVW = np.array(first_fe_solver.Solve(K_matrix[eval_id],jnp.ones(3*fe_mesh.GetNumberOfNodes())))  
fe_mesh['U_FE'] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

adj_fe_setting = {"linear_solver_settings":{"solver":"JAX-direct"}}
first_adj_fe_solver = AdjointFiniteElementSolver("first_adj_fe_solver",myresponse,adj_fe_setting)
first_adj_fe_solver.Initialize()

FE_adj_UVW = first_adj_fe_solver.Solve(K_matrix[eval_id],
                          FE_UVW,
                          jnp.ones(3*fe_mesh.GetNumberOfNodes()))

fe_mesh['ADJ_U_FE'] = FE_adj_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

# control_values = jnp.array(np.random.rand(fe_mesh.GetNumberOfNodes()))
# # jnp.ones((fe_mesh.GetNumberOfNodes()))
# dof_values = jnp.array(np.random.rand(3*fe_mesh.GetNumberOfNodes()))
# # print(dof_values.shape)
# # ll
# print(myresponse.ComputeValue(control_values,dof_values))

# myresponse.ComputeAdjointJacobianMatrixAndRHSVector(K_matrix[eval_id],FE_UVW)



# # solve FE here
# fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl"}}
# first_fe_solver = FiniteElementLinearResidualBasedSolver("first_fe_solver",mechanical_loss_3d,fe_setting)
# first_fe_solver.Initialize()
# FE_UVW = np.array(first_fe_solver.Solve(K_matrix[eval_id],jnp.ones(3*fe_mesh.GetNumberOfNodes())))  
# fe_mesh['U_FE'] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

fe_mesh.Finalize(export_dir=case_dir)


exit()




# design NN for learning
class MLP(nnx.Module):
    def __init__(self, in_features: int, dmid: int, out_features: int, *, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(in_features, dmid, rngs=rngs,kernel_init=nnx.initializers.zeros,bias_init=nnx.initializers.zeros)
        self.dense2 = nnx.Linear(dmid, out_features, rngs=rngs,kernel_init=nnx.initializers.zeros,bias_init=nnx.initializers.zeros)
        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.dense1(x)
        x = jax.nn.tanh(x)
        x = self.dense2(x)
        return x

fol_net = MLP(fourier_control.GetNumberOfVariables(),1, 
                mechanical_loss_3d.GetNumberOfUnknowns(), 
                rngs=nnx.Rngs(0))

# create fol optax-based optimizer
scheduler = optax.exponential_decay(
    init_value=1e-3,
    transition_steps=10,
    decay_rate=0.99)
chained_transform = optax.chain(optax.normalize_by_update_norm(), 
                                optax.scale_by_adam(),
                                optax.scale_by_schedule(scheduler),
                                optax.scale(-1.0))

# create fol
fol = ExplicitParametricOperatorLearning(name="dis_fol",control=fourier_control,
                                            loss_function=mechanical_loss_3d,
                                            flax_neural_network=fol_net,
                                            optax_optimizer=chained_transform,
                                            checkpoint_settings={"restore_state":False,
                                        "state_directory":f"./{working_directory_name}/flax_state"},
                                            working_directory=case_dir)

fol.Initialize()
eval_id = 1
# single sample training for eval_id
fol.Train(train_set=(coeffs_matrix[eval_id].reshape(-1,1).T,),
            convergence_settings={"num_epochs":200})

# predict for all samples 
FOL_UVWs = fol.Predict(coeffs_matrix)

# assign the prediction for  eval_id
fe_mesh['U_FOL'] = FOL_UVWs[eval_id].reshape((fe_mesh.GetNumberOfNodes(), 3))

# solve FE here
fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl"}}
first_fe_solver = FiniteElementLinearResidualBasedSolver("first_fe_solver",mechanical_loss_3d,fe_setting)
first_fe_solver.Initialize()
FE_UVW = np.array(first_fe_solver.Solve(K_matrix[eval_id],jnp.ones(3*fe_mesh.GetNumberOfNodes())))  
fe_mesh['U_FE'] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

fe_mesh.Finalize(export_dir=case_dir)
