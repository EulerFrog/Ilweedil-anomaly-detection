
import os
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



        # Parse and extract model parameters from 'config.yaml'
        yaml = open(self.model_path + "/config.yaml", "r")
        line = yaml.readline()
        line = yaml.readline()
        line_tokens = []
        while (line != ""):

            line = line.rstrip()
            line_tokens = line.split(": ")

            match (line_tokens[0]):

                case 'batch_size':
                    self.batch_size = float(line_tokens[1])
                case 'device':
                    self.device = line_tokens[1]
                case 'epochs':
                    self.epochs = int(line_tokens[1])
                case 'input_size':
                    self.input_size = int(line_tokens[1])
                case 'latent_size':
                    self.latent_size = int(line_tokens[1])
                case 'lr':
                    self.lr = float(line_tokens[1])
                case 'no_progress_bar':
                    self.no_progress_bar = bool(line_tokens[1])   
                case 'num_resamples':
                    self.num_resamples = int(line_tokens[1])       
                case 'steps_log_loss':
                    self.steps_log_loss = float(line_tokens[1])           
                case 'steps_log_norm_params':
                    self.steps_log_norm_params = float(line_tokens[1])    

            line = yaml.readline()

        yaml.close()

        # Load model last checkpoint by default
        self.LoadModel()
    

    def LoadModel(self, epoch: int=-1) -> int:
        """
            Loads an epoch of the model based on the inputted epoch. 
            Loaded model stored on the 'Model' property of this object

            Args:
                epoch - int # The epoch to load. Defaults to -1 (meaning it loads 'last' checkpoint)
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
        
        # If epoch is -1, load last checkpoint
        if (epoch == -1):

            if (os.path.exists(self.model_path + "/checkpoints/last.ckpt")):
                self.model = VAEAnomalyTabular.load_from_checkpoint(checkpoint_path=(self.model_path + "/checkpoints/last.ckpt"))
                return 1
            else:
                return 2
            
        # Otherwise, parse files for intended epoch
        else:

            for file in os.listdir(self.model_path + "/checkpoints/"):
                
                if (("epoch=" + str(epoch)) in file):
                    self.model = VAEAnomalyTabular.load_from_checkpoint(checkpoint_path=(self.model_path +  + "/checkpoints/" + file))
                    return 1
                
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


        
