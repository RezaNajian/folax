import os,time,sys
import numpy as np
import optax
from flax import nnx
import jax
from fol.mesh_input_output.mesh import Mesh
from fol.loss_functions.thermal import ThermalLoss3DTetra
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.controls.fourier_control import FourierControl
from fol.responses.fe_response import FiniteElementResponse
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.solvers.adjoint_fe_solver import AdjointFiniteElementSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
from fol.tools.decoration_functions import *
from functools import partial
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle,optax

class MetaAlphaMetaImplicitParametricOperatorLearningWithGrads(MetaAlphaMetaImplicitParametricOperatorLearning):
    @partial(nnx.jit, static_argnums=(0,))
    def ComputeBatchLatentX(self,batch_X:jnp.ndarray,flax_neural_network:nnx.Module,latent_step:nnx.Module):
        @nnx.jit
        def compute_single_latent(sample_x:jnp.ndarray):
            latent_code = jnp.zeros(flax_neural_network.in_features)
            control_output = sample_x
            @nnx.jit
            def loss(input_latent_code):
                nn_output = flax_neural_network(input_latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
                return self.loss_function.ComputeSingleLoss(control_output,nn_output)[0]
            loss_latent_grad_fn = jax.grad(loss)
            for _ in range(self.num_latent_iterations):
                latent_code -= latent_step() * loss_latent_grad_fn(latent_code)
            return latent_code
        return jnp.array(jax.vmap(compute_single_latent)(batch_X))
    @print_with_timestamp_and_execution_time
    def ComputeResponseValueAndGrad(self,sample_X:jnp.ndarray,response_function:FiniteElementResponse):
        @nnx.jit
        def compute_response_val(sample_X):
            latent_codes = self.ComputeBatchLatentX(sample_X,self.flax_neural_network,self.latent_step_nnx_model)
            batch_Y =jax.vmap(self.flax_neural_network,(0,None))(latent_codes,self.loss_function.fe_mesh.GetNodesCoordinates())
            batch_Y = batch_Y.reshape(latent_codes.shape[0], -1)[:,self.loss_function.non_dirichlet_indices]
            batch_Y = jax.vmap(self.loss_function.GetFullDofVector)(sample_X,batch_Y)
            return response_function.ComputeValue(sample_X.flatten(),batch_Y.flatten())
        return nnx.value_and_grad(compute_response_val,argnums=0) (sample_X)

# cleaning & logging
working_directory_name = 'thermal_ifol'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,"fol_thermal_3D.log"))

# create mesh_io
fe_mesh = Mesh("fol_io","fol_3D_tet_mesh_coarse.med",'../../meshes/')

# create fe-based loss function
bc_dict = {"T":{"left_fol":1,"right_fol":0.1}}

thermal_loss_3d = ThermalLoss3DTetra("thermal_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                        "beta":2,"c":4},
                                                                        fe_mesh=fe_mesh)

# create Fourier parametrization/control
x_freqs = np.array([1,2,3])
y_freqs = np.array([1,2,3])
z_freqs = np.array([0])
fourier_control_settings = {"x_freqs":x_freqs,"y_freqs":y_freqs,"z_freqs":z_freqs,"beta":5,"min":1e-1,"max":1}
fourier_control = FourierControl("K",fourier_control_settings,fe_mesh)
myresponse = FiniteElementResponse("my_response",response_formula="T",fe_loss=thermal_loss_3d,control=fourier_control)

fe_mesh.Initialize()
thermal_loss_3d.Initialize()
fourier_control.Initialize()
myresponse.Initialize()

# create some random coefficients & K for training
create_random_coefficients = False
if create_random_coefficients:
    number_of_random_samples = 200
    coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
    export_dict = {}
    export_dict["coeffs_matrix"] = coeffs_matrix
    export_dict["x_freqs"] = x_freqs
    export_dict["y_freqs"] = y_freqs
    export_dict["z_freqs"] = z_freqs
    with open(f'fourier_control_dict.pkl', 'wb') as f:
        pickle.dump(export_dict,f)
else:
    with open(f'fourier_control_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    coeffs_matrix = loaded_dict["coeffs_matrix"]
    K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

# now save K matrix 
export_Ks = False
if export_Ks:
    for i in range(K_matrix.shape[0]):
        fe_mesh[f'K_{i}'] = np.array(K_matrix[i,:])
    fe_mesh.Finalize(export_dir=case_dir)

eval_id = 5
fe_mesh['K'] = np.array(K_matrix[eval_id,:])


# now create implicit parametric deep learning
# design synthesizer & modulator NN for hypernetwork
characteristic_length = 64
synthesizer_nn = MLP(name="synthesizer_nn",
                     input_size=3,
                     output_size=1,
                     hidden_layers=[characteristic_length] * 6,
                     activation_settings={"type":"sin",
                                          "prediction_gain":30,
                                          "initialization_gain":1.0})

latent_size = characteristic_length
modulator_nn = MLP(name="modulator_nn",
                   input_size=latent_size,
                   use_bias=False) 

hyper_network = HyperNetwork(name="hyper_nn",
                             modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                             coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

# create fol optax-based optimizer
num_epochs = 2000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
main_loop_transform = optax.chain(optax.adam(1e-5))
latent_step_optimizer = optax.chain(optax.adam(1e-4))
# create fol
ifol = MetaAlphaMetaImplicitParametricOperatorLearningWithGrads(name="meta_implicit_ol",
                                                                control=fourier_control,
                                                                loss_function=thermal_loss_3d,
                                                                flax_neural_network=hyper_network,
                                                                main_loop_optax_optimizer=main_loop_transform,
                                                                latent_step_optax_optimizer=latent_step_optimizer,
                                                                latent_step_size=0.01)

ifol.Initialize()

ifol.Train(train_set=(coeffs_matrix[eval_id].reshape(-1,1).T,),
          convergence_settings={"num_epochs":2000,"relative_error":1e-100,"absolute_error":1e-100},
          plot_settings={"save_frequency":100},
          train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":1000,"state_directory":case_dir+"/flax_train_state"},
          test_checkpoint_settings={"least_loss_checkpointing":False,"frequency":1000,"state_directory":case_dir+"/flax_test_state"},
          restore_nnx_state_settings={"restore":False,"state_directory":case_dir+"/flax_final_state"},
          working_directory=case_dir)

# load the best model
ifol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

iFOL_T = np.array(ifol.Predict(coeffs_matrix[eval_id].reshape(-1,1).T)).reshape(-1)
fe_mesh['T_iFOL'] = iFOL_T.reshape((fe_mesh.GetNumberOfNodes(), 1))
Fol_res,FOL_res_sens = ifol.ComputeResponseValueAndGrad(K_matrix[eval_id].reshape(-1,1).T,myresponse)
fol_info(f"iFOL response value:{Fol_res}")
fe_mesh['FOL_sens'] = np.array(FOL_res_sens).reshape((fe_mesh.GetNumberOfNodes(), 1))

# solve FE here
fe_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                            "maxiter":20,"load_incr":1}}
nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",thermal_loss_3d,fe_setting)
nonlin_fe_solver.Initialize()
FE_T = nonlin_fe_solver.Solve(K_matrix[eval_id],np.zeros(fe_mesh.GetNumberOfNodes()))
fe_mesh['T_FE'] = np.array(FE_T.reshape((fe_mesh.GetNumberOfNodes(), 1)))
fol_info(f"FE response value:{myresponse.ComputeValue(K_matrix[eval_id],FE_T)}")

adj_fe_setting = {"linear_solver_settings":{"solver":"JAX-direct"}}
first_adj_fe_solver = AdjointFiniteElementSolver("first_adj_fe_solver",myresponse,adj_fe_setting)
first_adj_fe_solver.Initialize()

FE_adj_T = first_adj_fe_solver.Solve(K_matrix[eval_id],FE_T,jnp.ones(fe_mesh.GetNumberOfNodes()))
fe_mesh['adj_T_FE'] = np.array(FE_adj_T).reshape((fe_mesh.GetNumberOfNodes(), 1))

control_derivatives = myresponse.ComputeAdjointNodalControlDerivatives(K_matrix[eval_id],FE_T,FE_adj_T)
fe_mesh['FE_sens'] = np.array(control_derivatives).reshape((fe_mesh.GetNumberOfNodes(), 1))


# FD_control_sens = myresponse.ComputeFDNodalControlDerivatives(K_matrix[eval_id],nonlin_fe_solver,
#                                                     fd_step_size=1e-5,fd_mode="CD")

# fe_mesh['FD_sens'] = FD_control_sens.reshape((fe_mesh.GetNumberOfNodes(), 1))

relative_error = 100 * (abs(iFOL_T.reshape(-1,1)-FE_T.reshape(-1,1)))/abs(FE_T.reshape(-1,1))
fe_mesh['relative_error'] = relative_error.reshape((fe_mesh.GetNumberOfNodes(), 1))

fe_mesh.Finalize(export_dir=case_dir)
