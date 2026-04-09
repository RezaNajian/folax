import sys
import os
import numpy as np
import pickle 
from flax import nnx 
from functools import partial
import matplotlib.pyplot as plt
import jax.numpy as jnp  
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from categorical_deep_learning import CategoricalDeepLearning
from fol.controls.identity_control import IdentityControl
from fol.deep_neural_networks.nns import MLP
import pickle,optax
from fol.tools.decoration_functions import *


def main(num_epochs=10,clean_dir=False):

  # directory & save handling
  working_directory_name = "MNIST_training"
  case_dir = os.path.join('.', working_directory_name)
  create_clean_directory(working_directory_name)
  sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

  pre_process_training_data = False
  if pre_process_training_data:
    import tensorflow_datasets as tfds  # TFDS to download MNIST.
    import tensorflow as tf  # TensorFlow / `tf.data` operations.
    train_ds = tfds.as_numpy(tfds.load('mnist', split='train'))
    test_ds = tfds.as_numpy(tfds.load('mnist', split='test'))

    data_dict = {"train_images":[],
                "train_labels":[],
                "test_images":[],
                "test_labels":[]}

    for ex in train_ds:
      data_dict["train_images"].append(ex['image']/255)
      data_dict["train_labels"].append(ex['label'])

    data_dict["train_images"] = np.array(data_dict["train_images"])[0:1000]
    data_dict["train_labels"] = np.array(data_dict["train_labels"])[0:1000]

    for ex in test_ds:
        data_dict["test_images"].append(ex['image']/255)
        data_dict["test_labels"].append(ex['label'])

    data_dict["test_images"] = np.array(data_dict["test_images"])[0:1000]
    data_dict["test_labels"] = np.array(data_dict["test_labels"])[0:1000]

    print(f"Train Images: {data_dict['train_images'].shape}, Train Labels: {data_dict['train_labels'].shape}")
    print(f"Test Images: {data_dict['test_images'].shape}, Test Labels: {data_dict['test_labels'].shape}")

    with open(f'MNIST_data_dict.pkl', 'wb') as f:
        pickle.dump(data_dict,f)

  with open(f'MNIST_data_dict.pkl', 'rb') as f:
      data_dict = pickle.load(f)


  class CNN(nnx.Module):
    """A CNN model."""

    def __init__(self, *, rngs: nnx.Rngs):
      self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
      self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
      self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
      self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
      self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x):
      x = self.avg_pool(nnx.relu(self.conv1(x)))
      x = self.avg_pool(nnx.relu(self.conv2(x)))
      x = x.reshape(x.shape[0], -1)  # flatten
      x = nnx.relu(self.linear1(x))
      x = self.linear2(x)
      return x
    
  # Instantiate the model.
  cnn_network = True
  if cnn_network:
    deep_network = CNN(rngs=nnx.Rngs(0))
  else:
    deep_network = MLP("mlp",input_size=28*28,output_size=10,hidden_layers=[50,50],activation_settings={"type":"relu"})
    data_dict["train_images"] = data_dict["train_images"].reshape(data_dict["train_images"].shape[0], -1)
    data_dict["test_images"] = data_dict["test_images"].reshape(data_dict["test_images"].shape[0], -1)

  main_loop_transform = optax.chain(optax.adam(1e-5))
  ca_learning = CategoricalDeepLearning(name="mnist_categorical_learning",
                                        flax_neural_network=deep_network,
                                        optax_optimizer=main_loop_transform)

  ca_learning.Initialize()

  start_train_id = 0
  end_train_id = 1000
  start_test_id = 0
  end_test_id = 100
  ca_learning.Train(train_set=(data_dict["train_images"][start_train_id:end_train_id],
                              data_dict["train_labels"][start_train_id:end_train_id].reshape(-1,1)),
                    test_set=(data_dict["test_images"][start_test_id:end_test_id],data_dict["test_labels"][start_test_id:end_test_id].reshape(-1,1)),
                    test_frequency=10,
                    batch_size=10,
                    convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
                    plot_settings={"save_frequency":10},
                    train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":10},
                    working_directory=case_dir)

  train_set_predictions = ca_learning.Predict(data_dict["train_images"][start_train_id:end_train_id])
  train_set_predicted_cat = jnp.argmax(train_set_predictions,axis=1)
  train_set_error = abs(train_set_predicted_cat-data_dict['train_labels'][start_train_id:end_train_id])
  train_set_indices = np.nonzero(train_set_error)[0]
  fol_info(f"train_set_size:{train_set_predictions.shape}, num_wrong_preds:{train_set_indices.shape}")
  if train_set_indices.shape[0]>0:
    fig, axs = plt.subplots(5, 5, figsize=(12, 12))
    for i, ax in enumerate(axs.flatten()):
      ax.imshow(data_dict["train_images"][start_train_id:end_train_id][train_set_indices[i]].reshape(28,28,1), cmap='gray')
      ax.set_title(f'gt={data_dict["train_labels"][start_train_id:end_train_id][train_set_indices[i]]},pr={train_set_predicted_cat[train_set_indices[i]]}')
      ax.axis('off')
    fig.suptitle('Failed Train Set Predictions', fontsize=16)
    plt.savefig(case_dir+"/failed_train_set_predictions.png")

  test_set_predictions = ca_learning.Predict(data_dict["test_images"][start_test_id:end_test_id])
  test_set_predicted_cat = jnp.argmax(test_set_predictions,axis=1)
  test_set_error = abs(test_set_predicted_cat-data_dict['test_labels'][start_test_id:end_test_id])
  test_set_indices = np.nonzero(test_set_error)[0]
  fol_info(f"test_set_size:{test_set_predictions.shape}, num_wrong_preds:{test_set_indices.shape}")

  if test_set_indices.shape[0]>0:
    fig, axs = plt.subplots(5, 5, figsize=(12, 12))
    for i, ax in enumerate(axs.flatten()):
      ax.imshow(data_dict["test_images"][start_test_id:end_test_id][test_set_indices[i]].reshape(28,28,1), cmap='gray')
      ax.set_title(f'gt={data_dict["test_labels"][start_test_id:end_test_id][test_set_indices[i]]},pr={test_set_predicted_cat[test_set_indices[i]]}')
      ax.axis('off')
    fig.suptitle('Failed Test Set Predictions', fontsize=16)
    plt.savefig(case_dir+"/failed_test_set_predictions.png")

  if clean_dir:
    shutil.rmtree(case_dir)


if __name__ == "__main__":
    # Initialize default values
    num_epochs = 200
    clean_dir = False

    # Parse the command-line arguments
    args = sys.argv[1:]

    # Process the arguments if provided
    for arg in args:
        if arg.startswith("num_epochs="):
            try:
                num_epochs = int(arg.split("=")[1])
            except ValueError:
                print("num_epochs should be an integer.")
                sys.exit(1)
        elif arg.startswith("clean_dir="):
            value = arg.split("=")[1]
            if value.lower() in ['true', 'false']:
                clean_dir = value.lower() == 'true'
            else:
                print("clean_dir should be True or False.")
                sys.exit(1)
        else:
            print("Usage: python mnist_classifier.py num_epochs=10 clean_dir=False")
            sys.exit(1)

    # Call the main function with the parsed values
    main(num_epochs, clean_dir)