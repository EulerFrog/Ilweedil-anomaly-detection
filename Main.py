
import ModelInfo
from Download import download_netflow_dataset, stat_netflow_dataset
from Dataset import VAEDataset
from Pipeline import train_test
from ModelInfo import ModelInfo
import os

def main():

    # Load dataset 
    dataset = VAEDataset(from_file=True, data_file_path=os.getcwd() + "\\data\\netflow_data.csv")

    # Train and test model on dataset
    train_test(
        dataset=dataset,
        train_benign_dataset_size_percentage=0.6,
        train_anomaly_dataset_size_percentage=0.2,
        test_benign_dataset_size_percentage=0.05,
        test_anomaly_dataset_size_percentage=0.6
        )

    # model = ModelInfo(
    #     "test new train_test and netflow data",
    #     39
    # )

    # model.MassTestModel(
    #     dataset,
    #     1000,
    #     1000,
    #     100,
    #     "./",
    #     "verify results",
    #     -200  
    # )

if __name__ == '__main__':
    main()
