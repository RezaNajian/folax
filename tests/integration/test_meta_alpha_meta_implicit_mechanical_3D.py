import pytest
import unittest
import os
import numpy as np
import jax
import jax.numpy as jnp
import optax
from fol.mesh_input_output.mesh import Mesh
from fol.loss_functions.mechanical import MechanicalLoss3DTetra
from fol.controls.identity_control import IdentityControl
from fol.deep_neural_networks.nns import MLP, HyperNetwork
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import create_clean_directory

class TestMetaAlphaMetaImplicitMechanical3D(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _request_debug_mode(self, request):
        self.debug_mode = request.config.getoption('--debug-mode')

    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.join(os.path.dirname(__file__), 'test_meta_alpha_meta_implicit_mech3d')
        create_clean_directory(cls.test_dir)

    def setUp(self):
        # Mesh and mechanical loss setup
        mesh_dir = os.path.join(os.path.dirname(__file__), '../meshes')
        self.fe_mesh = Mesh("box_3D", "Box_3D_Tetra_20.med", mesh_dir)
        self.fe_mesh.Initialize()

        bc_dict = {
            "Ux": {"left": 0.0},
            "Uy": {"left": 0.0, "right": -0.05},
            "Uz": {"left": 0.0, "right": -0.05}
        }
        material_dict = {"young_modulus": 1.0, "poisson_ratio": 0.30}
        self.loss = MechanicalLoss3DTetra(
            "mech_loss",
            loss_settings={
                "dirichlet_bc_dict": bc_dict,
                "material_dict": material_dict,
                "body_force": jnp.zeros((3, 1))
            },
            fe_mesh=self.fe_mesh
        )
        self.loss.Initialize()

        # Identity control
        num_nodes = self.fe_mesh.GetNumberOfNodes()
        self.control = IdentityControl("identity_control", num_vars=num_nodes)
        self.control.Initialize()

        # Hypernetwork architecture
        synthesizer = MLP(
            "synth",
            input_size=3,
            output_size=3,
            hidden_layers=[16, 16],
            activation_settings={"type": "sin", "prediction_gain": 1.0, "initialization_gain": 1.0}
        )
        modulator = MLP("modulator", input_size=16 * 2, use_bias=False)
        self.hyper_net = HyperNetwork(
            "hyper_net",
            modulator,
            synthesizer,
            coupling_settings={
                "modulator_to_synthesizer_coupling_mode": "one_modulator_per_synthesizer_layer"
            }
        )

        # Meta‑learning object
        main_opt = optax.chain(
            optax.normalize_by_update_norm(),
            optax.adam(1e-4)
        )
        latent_opt = optax.chain(
            optax.normalize_by_update_norm(),
            optax.adamw(1e-5)
        )
        self.fol = MetaAlphaMetaImplicitParametricOperatorLearning(
            name="meta_implicit_mech3d",
            control=self.control,
            loss_function=self.loss,
            flax_neural_network=self.hyper_net,
            main_loop_optax_optimizer=main_opt,
            latent_step_optax_optimizer=latent_opt,
            latent_step_size=1e-3
        )
        self.fol.Initialize()

        # Create a single sample K_matrix
        # For identity control, we can use a uniform sample
        self.K_matrix = np.clip(np.ones((1, num_nodes)), 1e-2, 1.0)

    def test_meta_alpha_training_and_predict(self):
        # Run a tiny meta‑learning sanity check
        self.fol.Train(
            train_set=(self.K_matrix,),
            test_set=(self.K_matrix,),
            test_frequency=1,
            batch_size=1,
            convergence_settings={"num_epochs": 1},
            working_directory=self.test_dir
        )

        # Prediction should run and return correct shape
        pred = np.array(self.fol.Predict(self.K_matrix[0:1]))
        expected_dim = 3 * self.fe_mesh.GetNumberOfNodes()
        self.assertEqual(pred.shape, (1, expected_dim))

        # Cleanup unless debugging
        if self.debug_mode == "false":
            import shutil
            shutil.rmtree(self.test_dir)

