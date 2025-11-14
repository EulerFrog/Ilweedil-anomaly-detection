import argparse
from datetime import datetime
import os

from ModelInfo import ModelInfo
from Dataset import CSVDataset, VAEDataset
import torch
import yaml
from path import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.VAE import VAEAnomalyTabular
from torch.utils.data import Dataset, TensorDataset

ROOT = Path(__file__).parent
SAVED_MODELS = ROOT / 'saved_models'

def make_folder_run(name=None) -> Path:
    """
    Get the folder where to store the experiment.
    The folder is named with the current date and time.

    Returns:
        Path: the path to the folder where to store the experiment 
    """
    
    folder_name = ""
    if (name is None):
        folder_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    else:
        folder_name = name
    checkpoint_folder = SAVED_MODELS / folder_name
    checkpoint_folder.makedirs_p()
    return checkpoint_folder

def get_args() -> argparse.Namespace:
    """
    Parse command line arguments

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test-name', 
        type=str,
        required=True,
        dest='test_name',
        help='Name of the test'
    )
    parser.add_argument(
        '--latent-size', '-l',
        type=int,
        required=True,
        dest='latent_size',
        help='Size of the latent space'
    )
    parser.add_argument(
        '--num-resamples', '-L',
        type=int,
        dest='num_resamples',
        default=10,
        help='Number of resamples in the latent distribution during training'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        dest='epochs',
        default=100,
        help='Number of epochs to train for'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        dest='batch_size',
        default=32
    )
    parser.add_argument(
        '--device', '-d', '--accelerator',
        type=str,
        dest='device',
        default='cpu',
        help='Device to use for training. Can be cpu, gpu or tpu',
        choices=['cpu', 'gpu', 'tpu']
    )
    parser.add_argument(
        '--lr',
        type=float,
        dest='lr',
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--num-tests',
        type=int,
        dest='num_tests',
        default=5,
        help='Number of tests run during model testing'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        dest='alpha',
        default=0.5,
        help='Threshold for VAE when determining whether a record is an anomaly or not'
    )
    parser.add_argument(
        '--no-progress-bar',
        action='store_true',
        dest='no_progress_bar'
    )
    parser.add_argument(
        '--steps-log-loss',
        type=int,
        dest='steps_log_loss',
        default=1_000,
        help='Number of steps between each loss logging'
    )
    parser.add_argument(
        '--steps-log-norm-params',
        type=int,
        dest='steps_log_norm_params',
        default=1_000,
        help='Number of steps between each model parameters logging'
    )
    return parser.parse_args()

def train(args: dict, name_of_model: str, train_dataset:VAEDataset, validation_dataset: VAEDataset):
    """
    Main function to train the VAE model
    """
    print(args)

    experiment_folder = make_folder_run(name_of_model)


    with open(experiment_folder / 'config.yaml', 'w') as f:
        yaml.dump(args, f)

    checkpoint_folder = experiment_folder / 'checkpoints'
    checkpoint_folder.makedirs_p()



    model = VAEAnomalyTabular(
        train_dataset.__getinputsize__(),
        args.latent_size,
        args.num_resamples,
        lr=args.lr
    )


    train_dloader = DataLoader(train_dataset, args.batch_size, shuffle=True)

    val_dloader = DataLoader(validation_dataset, args.batch_size)

    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_folder,
        filename='{epoch:02d} {val_loss_epoch:.2f}',
        save_top_k=-1,
        verbose=True,
        monitor='val_loss_epoch',
        mode='min',
        save_last=True,
        every_n_epochs=1,
    )

    trainer = Trainer(
        callbacks=[checkpoint],
        max_epochs=args.epochs,
        accelerator=args.device,
        enable_progress_bar=not args.no_progress_bar,
    )

    trainer.fit(model, train_dloader, val_dloader)

    # When finished, return path to completed model at last checkpoint
    return 

def train_test(dataset: VAEDataset):
    """
        Trains and then tests a model based on args inputted to python. 

        Outputs the test to the "tests_folder" in the cwd.

        *In the case that tests share the same name, an increasing counter is added to the name of the test:
        e.g. (for test name of "test")
            test
            test 1
            test 2
            .
            .
            test n
    """

    # Locals
    args = None
    test_model_path = ""
    test_folder_name = os.getcwd() + '\\test_folder'
    str_holder = ""
    str_holder_2 = ""
    i = 1

    # Get arguments
    args = get_args()

    # Create "test_folder" if it doesn't exist already
    if not os.path.isdir(test_folder_name):
        os.mkdir(test_folder_name)

    # Within test_folder, check for models of the same name as this test. If they exist, create a new
    #   name for the model that isn't seen in the folder.
    test_model_path = test_folder_name + "\\" + args.test_name
    str_holder = test_model_path
    while os.path.isdir(str_holder):
        str_holder = test_model_path + " " + str(i)
        i = i + 1
    test_model_path = str_holder    
    if (i > 1):
        args.test_name = args.test_name + " " + str(i)

    # Train the model
    train(args, args.test_name, dataset, dataset)

    # Test the model
    model = ModelInfo(args.test_name, dataset.__getinputsize__())
    model.LoadModel()
    model.MassTestModel(
        dataset, 
        args.num_tests, 
        test_model_path, 
        args.test_name
        )


