
import os
import json
import torch
from pathlib import Path
from dataset import CSVDataset
from model.VAE import VAEAnomalyTabular
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


"""
    Class that provides information about models created.
        - Can generate paths to a model and particular epochs within that model.
        - Can also provide information on the hyperparameters used when training a model.
"""
class ModelInfo:

    """
        ModelInfo constructor. 

        Takes a model_id as an input, representing the name of the folder containing the model
        in "saved_models"
    """
    def __init__(
        self,
        model_id: str 
    ):
        
        # Define fields
        self.model = None
        self.model_path = ""
        self.batch_size = None
        self.device = None
        self.epochs = None
        self.input_size = None
        self.latent_size = None
        self.lr = None
        self.no_progress_bar = None
        self.num_resamples = None
        self.steps_log_loss = None
        self.steps_log_norm_params = None


        # Search for 'saved_models' in cwd. If found, continue
        for file in os.listdir('.'):
            if (file == "saved_models"):
                for file2 in os.listdir('./saved_models'):
                    if (file2 == model_id):
                        self.model_path = os.getcwd() + '/saved_models/' + model_id



        # Parse and extract model parameters from 'config.json'
        with open(self.model_path + "/config.json", "r") as f:
            config = json.load(f)

        # Extract configuration parameters
        self.batch_size = config.get('batch_size')
        self.device = config.get('device')
        self.epochs = config.get('epochs')
        self.input_size = config.get('input_size')
        self.latent_size = config.get('latent_size')
        self.lr = config.get('lr')
        self.no_progress_bar = config.get('no_progress_bar')
        self.num_resamples = config.get('num_resamples')
        self.steps_log_loss = config.get('steps_log_loss')
        self.steps_log_norm_params = config.get('steps_log_norm_params')

        # Load model last checkpoint by default
        self.LoadModel()
    

    def LoadModel(self, epoch: int=-1, checkpoint_type: str='last') -> int:
        """
            Loads an epoch of the model based on the inputted epoch.
            Loaded model stored on the 'Model' property of this object

            Args:
                epoch - int # The epoch to load. Defaults to -1 (meaning it loads 'last' checkpoint)
                checkpoint_type - str # Type of checkpoint to load: 'last', 'best', or specific epoch number
            Returns:
                1 if successfully loaded
                2 if failed to load (file not found)
                3 if failed to load (model path not found)
                4 if failed to load (epoch out of bounds)
            *If model_path is not initialized, returns
        """

        # If model path not initialized, return
        if (self.model_path == ""):
            return 3

        # If epoch out of range, return
        if (epoch > self.epochs):
            return 4
        if (epoch == self.epochs):
            epoch = -1

        checkpoint_path = None

        # If epoch is -1, load last or best checkpoint based on checkpoint_type
        if (epoch == -1):
            if checkpoint_type == 'best' and os.path.exists(self.model_path + "/checkpoints/best.ckpt"):
                checkpoint_path = self.model_path + "/checkpoints/best.ckpt"
            elif os.path.exists(self.model_path + "/checkpoints/last.ckpt"):
                checkpoint_path = self.model_path + "/checkpoints/last.ckpt"
            else:
                return 2

        # Otherwise, parse files for intended epoch
        else:
            for file in os.listdir(self.model_path + "/checkpoints/"):
                if (("epoch_" + str(epoch).zfill(3)) in file):
                    checkpoint_path = self.model_path + "/checkpoints/" + file
                    break

            if checkpoint_path is None:
                return 2

        # Load the checkpoint
        try:
            checkpoint = torch.load(checkpoint_path)

            # Initialize model with saved config
            self.model = VAEAnomalyTabular(
                self.input_size,
                self.latent_size,
                self.num_resamples,
                lr=self.lr
            )

            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Set to evaluation mode

            return 1
        except Exception as e:
            print(f"Error loading model: {e}")
            return 2
            
                
    def TestModel(self, testDataset: CSVDataset, alpha:float = 0.5, testBatchSize:int = -1) -> dict:
        """
            This function tests a this model against the inputted dataset.

            Inputted dataset is expected to be of self.input_size + 1 with the last input
            representing a number either 0 or 1. 0 Represents a non-anomylous record while 1
            does.

            After completing testing, the results are placed in a .csv file within the cwd.

            Args:
                Dataset to use for testing
                    Expects dataset in form:
                    x = Tensor(
                        Tensor()
                        Tensor()
                        .
                        .
                        .
                        Tensor()
                    )
                    labels = Tensor(
                    )
                    Where x holds the tensors of data records and labels is the parallel tensor with labels of records in x
                    ranging from [0, 1] where 0 is normal and 1 is an anomaly

                Alpha (threshold for anomaly. Any above identified as an anomaly, any below doesn't)
                Batch size for testing (-1 indicates all values in dataset)

            Returns:
                1 if tested successfully
                2 otherwise
        """

        # If model isn't loaded, return
        if (self.model == None):
            return 2
        

        if (testBatchSize <= -1):
            testBatchSize = testDataset.__len__() - 1
        

        i = 0
        while (i < len(testDataset)):

            print(testDataset[i])
            print(testDataset.__getitemlabel__(i))
            print(self.model.is_anomaly(testDataset[i], alpha))
            input()


        
