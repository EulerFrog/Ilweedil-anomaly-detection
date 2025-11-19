import os
from pathlib import Path

import torch
from torch import Tensor
from Dataset import VAEDataset
from model.VAE import VAEAnomalyTabular
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ModelInfo:
    """
        Class that provides information about models created.
            - Can generate paths to a model and particular epochs within that model.
            - Can also provide information on the hyperparameters used when training a model.
    """
    # Fields used in testing
    test_fields = ["FN", "FP", "TN", "TP", "F1", "Accuracy", "Precision", "Recall", "F1"]

    
    def __init__(
        self,
        model_id: str,
        model_input_size: int 
    ):
        """
            ModelInfo constructor. 

            Takes a model_id as an input, representing the name of the folder containing the model
            in "saved_models"
        """    
        # Define fields
        self.model_id = model_id
        self.model = None
        self.model_path = ""
        self.batch_size = None
        self.device = None
        self.epochs = None
        self.input_size = model_input_size
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

            if (line_tokens[0] == 'batch_size'):
                self.batch_size = float(line_tokens[1])
            elif (line_tokens[0] == 'device'):
                self.device = line_tokens[1]
            elif (line_tokens[0] == 'epochs'):
                self.epochs = int(line_tokens[1])
            elif (line_tokens[0] == 'latent_size'):
                self.latent_size = int(line_tokens[1])
            elif (line_tokens[0] == 'lr'):
                self.lr = float(line_tokens[1])
            elif (line_tokens[0] == 'no_progress_bar'):
                self.no_progress_bar = bool(line_tokens[1])
            elif (line_tokens[0] == 'num_resamples'):
                self.num_resamples = int(line_tokens[1])       
            elif (line_tokens[0] == 'steps_log_loss'):
                self.steps_log_loss = float(line_tokens[1])           
            elif (line_tokens[0] == 'steps_log_norm_params'):
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
                self.model = VAEAnomalyTabular(
                    self.input_size, 
                    self.latent_size, 
                    self.num_resamples, 
                    self.lr) 
                self.model.load_state_dict(torch.load(self.model_path + "/checkpoints/last.ckpt", weights_only=False)['model_state_dict'])
                return 1
            else:
                return 2
            
        # Otherwise, parse files for intended epoch
        else:

            for file in os.listdir(self.model_path + "/checkpoints/"):
                
                if (("epoch=" + str(epoch)) in file):
                    self.model = VAEAnomalyTabular(
                        self.input_size, 
                        self.latent_size, 
                        self.num_resamples, 
                        self.lr) 
                    self.model.load_state_dict(torch.load(self.model_path +  + "/checkpoints/" + file, weights_only=False)['model_state_dict']
)
                    return 1
                
            return 2
            
                
    def MassTestModel(self, dataset: VAEDataset, test_rounds:int , results_output_path: str, test_name: str, alpha: float = 0.5):
        """
            Runs "test_rounds" tests on the model using "dataset" to do so. Outputs the results as a .csv file 
            to "results_output_path".
        """

        # Locals
        results = []
        batch = None
        data_records = []
        data_record_labels = []
        results_aggregates = dict()
        test_data_loader = None

        # Initialize test results dictionary
        for field in ModelInfo.test_fields:
            results_aggregates[field] = {
                "max":-1,
                "min":-1,
                "avg":0
            }

        # Initialize test_data_loader as dataloader of the remaining data records in the VAEDataset
        remaining_anomalous_data = dataset.anomalous_data_length - dataset.unallocated_anomalous_data_start_index
        remaining_benign_data = dataset.benign_data_length - dataset.unallocated_benign_data_start_index
        test_data_loader = dataset.get_dataloader(
            self.batch_size,
            benign_size=remaining_benign_data,
            anomalous_size=remaining_anomalous_data
        )

        # Perform tests. For each test round, do a test of size 'batch_size'
        i = 0
        for data_records, data_record_labels in test_data_loader:            

            results.append(self.TestModel(data_records, data_record_labels))

            i = i + 1
            if (i >= test_rounds):
                break

        # Calculate aggregate of test results
        for test in results:

            if (test != {}):
                for field in ModelInfo.test_fields:

                    # Calculate min
                    if (test[field] < results_aggregates[field]["min"] or results_aggregates[field]["min"] == -1):
                        results_aggregates[field]["min"] = test[field]

                    # Calculate max
                    if (test[field] > results_aggregates[field]["max"] or results_aggregates[field]["max"] == -1):
                        results_aggregates[field]["max"] = test[field]

                    # Store sum of values in 'avg' to be divided later
                    if (test[field] != -1):
                        results_aggregates[field]["avg"] = results_aggregates[field]["avg"] + test[field]
        
        #   Calculate avg
        for field in ModelInfo.test_fields:

            if (results_aggregates[field]["avg"] != -1):
                results_aggregates[field]["avg"] = results_aggregates[field]["avg"] / len(results)

            # print(str(results_aggregates[field]["avg"]))

        # Write test results to file
        with open(results_output_path + ".csv", "w+") as output_file:

            # Write name of test
            output_file.write(test_name + ',\n')

            # Write number of rounds
            output_file.write("Test rounds:," + str(test_rounds) + ',\n')

            # Write hyper parameters of model
            #   Header for hyper parameters
            output_file.write("Batch size, Latent size, Input size, Learning rate, Epochs, Num resamples, Alpha,\n")
            #   Hyper parameters
            output_file.write(str(self.batch_size) + ",") # batch size
            output_file.write(str(self.latent_size) + ",") # latent size
            output_file.write(str(self.input_size) + ",") # input size
            output_file.write(str(self.lr) + ",") # lr
            output_file.write(str(self.epochs) + ",") # epochs
            output_file.write(str(self.num_resamples) + ",") # num resamples
            output_file.write(str(alpha) + ",\n") # alpha
            

            # Write aggregate of test results
            #   Header
            output_file.write("\n")
            output_file.write(" ,")
            for field in ModelInfo.test_fields:
                output_file.write(field + ",")
            output_file.write("\n")

            #   Avg
            output_file.write("Avg,")
            for field in ModelInfo.test_fields:
                output_file.write(str(results_aggregates[field]["avg"]) + ",") 
            output_file.write("\n")

            #   Min
            output_file.write("Min,")
            for field in ModelInfo.test_fields:
                output_file.write(str(results_aggregates[field]["min"]) + ",") 
            output_file.write("\n")

            #   Max
            output_file.write("Max,")
            for field in ModelInfo.test_fields:
                output_file.write(str(results_aggregates[field]["min"]) + ",") 
            output_file.write("\n")


            # Write individual test results
            output_file.write(",\n")
            i = 1
            for result in results:
                output_file.write("Test " + str(i) + ",")
                for field in ModelInfo.test_fields:
                    output_file.write(str(result[field]) + ",") 
                output_file.write('\n')
                i = i + 1

            # Close file and return
            output_file.close()
            

    def TestModel(self, test_dataset_records: Tensor, test_dataset_labels: Tensor, alpha:float = 0.5) -> dict:
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
                        [data record 1]
                        [data record 2]
                        .
                        .
                        .
                        [data record n]
                    )
                    labels = Tensor(
                    )
                    Where x holds the tensors of data records and labels is the parallel tensor with labels of records in x
                    ranging from [0, 1] where 0 is normal and 1 is an anomaly

                Labels used for testing
                    y = Tensor([
                        label of data record 1,
                        label of data record 2,
                        .
                        .
                        .
                        label of data record n
                    ])
                Alpha (threshold for anomaly. Any above identified as an anomaly, any below doesn't)

            Returns:
                If successful: A dictionary of values based on the result of testing in the format of:
                {
                    # Hyper parameters
                    "batch_size":self.batch_size, 
                    "latent_size":self.latent_size,
                    "input_size":self.input_size,
                    "lr":self.lr,
                    "epochs":self.epochs,
                    "num_resamples":self.num_resamples,
                    "alpha":alpha

                    # Results
                    "TP":{int},
                    "TN":{int},
                    "FP":{int},
                    "FN":{int},
                    "Accuracy":{float},
                    "Precision":{float},
                    "Recall":{float},
                    "F1":{float}
                }
                *Note, the values of Precision, Recall, and/or F1 may not be able to be calculated depending on FN, FP, TP, TN (div by 0). When
                this is the case, their values will be set to -1.

                If unsuccessful:
                {} 
        """

        # Locals
        return_dict = dict()
        result = None
        FP = 0
        TP = 0
        TN = 0
        FN = 0
        acc = 0 
        rec = 0
        prec = 0
        f1 = 0


        # If model isn't loaded, return
        if (self.model == None):
            return {}
        

        # Test batch size is always equal to model hyper parameter "batch size"
        # Verify tensor is of correct size before testing (batch size x input size)
        test_dataset_records_size = test_dataset_records.size()
        test_dataset_labels_size = test_dataset_labels.size()
        if (test_dataset_records_size[0] != self.batch_size or test_dataset_records_size[1] != self.input_size):
            print("Error - Could not test dataset.")
            print("For record tensor, expected size of: [" + str(self.batch_size) + "," + str(self.input_size) + "]")
            print("Got size of: " + str(test_dataset_records_size))
            return {}
        if (test_dataset_labels_size[0] != self.batch_size):
            print("Error - Could not test dataset.")
            print("For label tensor, expected size of: [" + self.batch_size + "," + self.input_size + "]")
            print("Got size of: " + test_dataset_labels_size)
            return {}
        
        # Run tests
        result = self.model.is_anomaly(test_dataset_records, alpha)

        # Run calculations based on results.
        i = 0
        while (i < self.batch_size):

            # print('Batch index ' + str(i))
            # print(test_dataset_labels[i])
            # print(result[i])
            # print('***')
            if (test_dataset_labels[i] == 0 and result[i] == 0):
                TN = TN + 1
            elif (test_dataset_labels[i] == 1 and result[i] == 0):
                FN = FN + 1
            elif (test_dataset_labels[i] == 0 and result[i] == 1):
                FP = FP + 1
            elif (test_dataset_labels[i] == 1 and result[i] == 1):
                TP = TP + 1
            i = i + 1

        # print("Test results:")
        # print("TP = " + str(TP))
        # print("TN = " + str(TN))
        # print("FP = " + str(FP))
        # print("FN = " + str(FN))
        
        acc = ((TP + TN) / (self.batch_size))
        if (TP+FP == 0):
            prec = -1
        else:
            prec = (TP/(TP+FP))

        if (FP+TN == 0):
            rec = -1
        else:
            rec = (FP/(FP+TN))

        if ((prec == -1 or rec == -1) or (prec+rec == 0)):
            f1 = -1
        else:
            f1 = 2 * ((prec * rec)/(prec+rec))
        # print("Accuracy = " +  str(acc))
        # print("Precision = " + str(prec))
        # print("Recall = " + str(rec))
        # print("F1 = " + str(f1))

        
        # Pack up and return results

        #   Params
        return_dict["batch_size"] = self.batch_size
        return_dict["num_resamples"] = self.num_resamples
        return_dict["latent_size"] = self.latent_size
        return_dict["input_size"] = self.input_size
        return_dict["lr"] = self.lr
        return_dict["epochs"] = self.epochs
        return_dict["alpha"] = alpha

        #   Results
        return_dict["FN"] = FN
        return_dict["FP"] = FP
        return_dict["TP"] = TP
        return_dict["TN"] = TN
        return_dict["Accuracy"] = acc
        return_dict["Precision"] = prec
        return_dict["Recall"] = rec
        return_dict["F1"] = f1

        #   Return
        return return_dict
        
