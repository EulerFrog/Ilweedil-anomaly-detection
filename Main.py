"""
    Main.py
    Last modified: 12/3/25
    Description: 
        Holds code intended for main execution.
"""
# Imports
from Dataset import VAEDataset
from Pipeline import train_test
import os

def main():

    #
    #   
    #

    #
    #   Training and testing the model
    #
    # Load dataset 
    dataset = VAEDataset(from_file=True, data_file_path=os.getcwd() + "/data/data.csv")

    # Train and test model on dataset
    train_test(
        dataset=dataset,
        train_benign_dataset_size_percentage=0.05,
        train_anomaly_dataset_size_percentage=0.2,
        test_benign_dataset_size_percentage=0.05,
        test_anomaly_dataset_size_percentage=0.6
        )


if __name__ == '__main__':
    main()
