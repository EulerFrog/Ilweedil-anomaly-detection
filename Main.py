
import ModelInfo
from Download import download_netflow_dataset, stat_netflow_dataset
from Dataset import VAEDataset
from Pipeline import train_test
import os

def main():

    # Load dataset 
    dataset = VAEDataset(from_file=True, data_file_path=os.getcwd() + "/NIDS_cleaned_dataset.csv")

    # Train and test model on dataset
    train_test(dataset=dataset)

if __name__ == '__main__':
    main()
