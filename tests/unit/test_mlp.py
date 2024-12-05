import pytest
import unittest
import os
import numpy as np
from fol.deep_neural_networks.nns import MLP
from fol.tools.usefull_functions import *
import jax

class TestMLP(unittest.TestCase):

    def test_siren_mlp(self):

        relu_mlp = MLP(name="relu_mlp",
                     input_size=3,
                     output_size=1,
                     hidden_layers=[])
        self.assertEqual(relu_mlp.total_num_biases,1)
        self.assertEqual(relu_mlp.total_num_weights,3)

        relu_mlp = MLP(name="relu_mlp",
                     input_size=3,
                     output_size=1,
                     hidden_layers=[2,2,2],
                     activation_settings={"type":"relu"})
        self.assertEqual(relu_mlp.total_num_biases,7)
        self.assertEqual(relu_mlp.total_num_weights,16)
        self.assertEqual(relu_mlp.activation_settings,{"type":"relu",
                                                       "prediction_gain":30,
                                                       "initialization_gain":1})
        self.assertEqual(relu_mlp.skip_connections_settings,{"active":False,"frequency":1})
        self.assertEqual(relu_mlp.act_func,jax.jax.nn.relu)
        self.assertEqual(relu_mlp.act_func_gain,1)
        self.assertEqual(relu_mlp.fw_func,relu_mlp.Forward)

        siren_mlp = MLP(name="siren",
                     input_size=3,
                     output_size=1,
                     hidden_layers=[2,2,2],
                     activation_settings={"type":"sin",
                                          "prediction_gain":50,
                                          "initialization_gain":3},
                    skip_connections_settings={"active":True,"frequency":1})

        self.assertEqual(siren_mlp.total_num_biases,7)
        self.assertEqual(siren_mlp.total_num_weights,31)
        self.assertEqual(siren_mlp.activation_settings,{"type":"sin",
                                                        "prediction_gain":50,
                                                        "initialization_gain":3})
        self.assertEqual(siren_mlp.skip_connections_settings,{"active":True,"frequency":1})

        skip_siren_mlp = MLP(name="siren",
                     input_size=3,
                     output_size=2,
                     hidden_layers=[4,5,6,7],
                     activation_settings={"type":"sin",
                                          "prediction_gain":50,
                                          "initialization_gain":3},
                    skip_connections_settings={"active":True,"frequency":2})
        self.assertEqual(skip_siren_mlp.activation_settings,{"type":"sin",
                                                        "prediction_gain":50,
                                                        "initialization_gain":3})
        self.assertEqual(skip_siren_mlp.skip_connections_settings,{"active":True,"frequency":2})
        self.assertEqual(skip_siren_mlp.fw_func,skip_siren_mlp.ForwardSkip)
        self.assertEqual(skip_siren_mlp.total_num_biases,24)
        self.assertEqual(skip_siren_mlp.total_num_weights,142)
        self.assertEqual(skip_siren_mlp.nn_params[0][0].shape,(3,4))
        self.assertEqual(skip_siren_mlp.nn_params[1][0].shape,(4,5))
        self.assertEqual(skip_siren_mlp.nn_params[2][0].shape,(8,6))
        self.assertEqual(skip_siren_mlp.nn_params[3][0].shape,(6,7))
        self.assertEqual(skip_siren_mlp.nn_params[4][0].shape,(10,2))
        self.assertEqual(skip_siren_mlp.act_func,jax.numpy.sin)
        self.assertEqual(skip_siren_mlp.act_func_gain,50)
        np.testing.assert_allclose(np.array(skip_siren_mlp(jnp.array([[1,2,3],[4,5,6],[7,8,9]]))).flatten(), 
                                    np.array([-0.06585833,-0.032813,-0.127911,-0.02603725,-0.20289356,-0.07241035]), rtol=1e-5, atol=1e-10)

if __name__ == '__main__':
    unittest.main()