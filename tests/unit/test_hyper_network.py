import pytest
import unittest
import os
import numpy as np
from fol.deep_neural_networks.nns import MLP,HyperNetwork
from fol.tools.usefull_functions import *
import jax

class TestHyperNetworks(unittest.TestCase):

    def test_hn_all_to_all(self):

        # first check all to all coupling
        with self.assertRaises(SystemExit):
            HyperNetwork(name="test_hypernetwork", 
                        modulator_nn=MLP(name="modulator_nn",input_size=20,output_size=1,hidden_layers=[1,2,3]),
                        synthesizer_nn=MLP(name="synthesizer_nn",input_size=3,output_size=1,hidden_layers=[4,5,6]))

        with self.assertRaises(SystemExit):
            HyperNetwork(name="test_hypernetwork", 
                        modulator_nn=MLP(name="modulator_nn",input_size=20,output_size=1,hidden_layers=[1,2,3]),
                        synthesizer_nn=MLP(name="synthesizer_nn",input_size=3,output_size=1,hidden_layers=[1,2,3]),
                        coupling_settings={"coupled_variable":"weight"})

        hyper_network = HyperNetwork(name="test_hypernetwork", 
                                    modulator_nn=MLP(name="modulator_nn",input_size=20,hidden_layers=[1,2,3],activation_settings={"type":"relu"}),
                                    synthesizer_nn=MLP(name="synthesizer_nn",input_size=3,output_size=1,hidden_layers=[1,2,3],activation_settings={"type":"sin"}))
        self.assertEqual(hyper_network.coupling_settings,{"coupled_variable":"shift",
                                                          "modulator_to_synthesizer_coupling_mode":"all_to_all"})
        self.assertEqual(hyper_network.total_num_biases,hyper_network.modulator_nn.total_num_biases+hyper_network.synthesizer_nn.total_num_biases)
        self.assertEqual(hyper_network.total_num_weights,hyper_network.modulator_nn.total_num_weights+hyper_network.synthesizer_nn.total_num_weights)
        self.assertEqual(hyper_network.fw_func,hyper_network.all_to_all_fw)

        latent_vector = jax.random.normal(jax.random.PRNGKey(41), shape=(20,))
        coord_matrix = jax.random.normal(jax.random.PRNGKey(41), shape=(10,3))

        np.testing.assert_allclose(np.array(hyper_network(latent_vector,coord_matrix)).flatten(), 
                            np.array([-0.07999684, -0.05501849, -0.07996142, -0.03240417, -0.04693764, -0.06308696,
                                        -0.02268538, -0.059735, -0.05208695, -0.03246102]), rtol=1e-5, atol=1e-10)

        hyper_network = HyperNetwork(name="test_hypernetwork", 
                                    modulator_nn=MLP(name="modulator_nn",input_size=20,hidden_layers=[1,2,3],activation_settings={"type":"relu"},skip_connections_settings={"active":True,"frequency":1}),
                                    synthesizer_nn=MLP(name="synthesizer_nn",input_size=3,output_size=1,hidden_layers=[1,2,3],activation_settings={"type":"sin"},skip_connections_settings={"active":True,"frequency":1}))

        np.testing.assert_allclose(np.array(hyper_network(latent_vector,coord_matrix)).flatten(), 
                            np.array([0.01455511,0.00062445,-0.01205211,-0.09482165,0.0424481,0.03400226,
                                      -0.01980503,0.03233872,0.03581288,0.0823221 ]), rtol=1e-5, atol=1e-10)

        hyper_network = HyperNetwork(name="test_hypernetwork", 
                                    modulator_nn=MLP(name="modulator_nn",input_size=20,hidden_layers=[2,2,2,2],activation_settings={"type":"relu"}),
                                    synthesizer_nn=MLP(name="synthesizer_nn",input_size=3,output_size=1,hidden_layers=[2,2,2,2],activation_settings={"type":"sin"},skip_connections_settings={"active":True,"frequency":2}))

        np.testing.assert_allclose(np.array(hyper_network(latent_vector,coord_matrix)).flatten(), 
                            np.array([-0.01560773,0.03158602,0.02431477,-0.04337462,-0.01519061,0.06405739,
                                        0.05395119,0.07285306,0.05532924,-0.03511214]), rtol=1e-5, atol=1e-10)

    def test_hn_last_to_all(self):
        hyper_network = HyperNetwork(name="test_hypernetwork", 
                                    modulator_nn=MLP(name="modulator_nn",input_size=20,output_size=1,hidden_layers=[1,2,3]),
                                    synthesizer_nn=MLP(name="synthesizer_nn",input_size=3,output_size=3,hidden_layers=[4,5,6],skip_connections_settings={"active":True,"frequency":1}),
                                    coupling_settings={"coupled_variable":"shift","modulator_to_synthesizer_coupling_mode":"last_to_all"})
        self.assertEqual(hyper_network.modulator_nn.out_features,hyper_network.synthesizer_nn.total_num_biases-hyper_network.synthesizer_nn.out_features)
        self.assertEqual(hyper_network.fw_func,hyper_network.last_to_all_fw)

        latent_vector = jax.random.normal(jax.random.PRNGKey(41), shape=(20,))
        coord_matrix = jax.random.normal(jax.random.PRNGKey(41), shape=(3,3))

        np.testing.assert_allclose(np.array(hyper_network(latent_vector,coord_matrix)).flatten(),np.array([0.05603085,-0.02487829,0.05327738,-0.0061287,0.04346357,-0.05731859,0.00075935,0.02638959,-0.04860119])
                                   ,rtol=1e-5, atol=1e-10)
        hyper_network = HyperNetwork(name="test_hypernetwork", 
                                    modulator_nn=MLP(name="modulator_nn",input_size=20,output_size=1,hidden_layers=[1,2,3],skip_connections_settings={"active":True,"frequency":2}),
                                    synthesizer_nn=MLP(name="synthesizer_nn",input_size=3,output_size=3,hidden_layers=[4,5,6],skip_connections_settings={"active":True,"frequency":1}),
                                    coupling_settings={"coupled_variable":"shift","modulator_to_synthesizer_coupling_mode":"last_to_all"})
        self.assertEqual(hyper_network.modulator_nn.out_features,hyper_network.synthesizer_nn.total_num_biases-hyper_network.synthesizer_nn.out_features)
        self.assertEqual(hyper_network.fw_func,hyper_network.last_to_all_fw)

        np.testing.assert_allclose(np.array(hyper_network(latent_vector,coord_matrix)).flatten(),np.array([-0.00213177,-0.02584138,0.05836014,-0.05818252,0.08189567,-0.05966015,-0.01929847,0.05629948,-0.07365533])
                                   ,rtol=1e-5, atol=1e-10)
        
    def test_hn_one_modulator_per_synthesizer_layer(self):
        hyper_network = HyperNetwork(name="test_hypernetwork", 
                                    modulator_nn=MLP(name="modulator_nn",input_size=20,hidden_layers=[5]),
                                    synthesizer_nn=MLP(name="synthesizer_nn",input_size=3,output_size=3,hidden_layers=[4,5,6],skip_connections_settings={"active":True,"frequency":1}),
                                    coupling_settings={"coupled_variable":"shift","modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

        self.assertEqual(hyper_network.total_num_biases,48)
        self.assertEqual(hyper_network.total_num_weights,497)
        self.assertEqual(len(hyper_network.modulator_nns),3)
        self.assertEqual(hyper_network.modulator_nns[0].hidden_layers,[5])
        self.assertEqual(hyper_network.modulator_nns[0].in_features,20)
        self.assertEqual(hyper_network.modulator_nns[0].out_features,4)
        self.assertEqual(hyper_network.modulator_nns[0].skip_connections_settings,{"active":False,"frequency":1})
        self.assertEqual(hyper_network.modulator_nns[1].out_features,5)
        self.assertEqual(hyper_network.modulator_nns[2].out_features,6)
        self.assertEqual(hyper_network.fw_func,hyper_network.one_modulator_per_synthesizer_layer_fw)
        latent_vector = jax.random.normal(jax.random.PRNGKey(41), shape=(20,))
        coord_matrix = jax.random.normal(jax.random.PRNGKey(41), shape=(3,3))
        
        np.testing.assert_allclose(np.array(hyper_network(latent_vector,coord_matrix)).flatten(),
                                   np.array([0.04693167,-0.06193891,0.07650414,0.00229242,0.04451895,-0.05770272,0.03079399,0.02711906,-0.05261766]),
                                   rtol=1e-5, atol=1e-10)

        hyper_network = HyperNetwork(name="test_hypernetwork", 
                                    modulator_nn=MLP(name="modulator_nn",input_size=20,hidden_layers=[]),
                                    synthesizer_nn=MLP(name="synthesizer_nn",input_size=3,output_size=3,hidden_layers=[4,5,6]),
                                    coupling_settings={"coupled_variable":"shift","modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

        self.assertEqual(hyper_network.total_num_biases,33)
        self.assertEqual(hyper_network.total_num_weights,380)
        self.assertEqual(len(hyper_network.modulator_nns),3)

        np.testing.assert_allclose(np.array(hyper_network(latent_vector,coord_matrix)).flatten(),
                                   np.array([-0.03324022,0.07827269,0.06702476,-0.00612662,0.01867586,-0.00902701,
                                             -0.00846323,0.02852342,-0.00236065]),
                                   rtol=1e-5, atol=1e-10)

if __name__ == '__main__':
    unittest.main()