
import ModelInfo
from Download import download_netflow_dataset
from Dataset import CSVDataset, NetflowDatset
from Utility import Train_Test
import os

def main():

    test = NetflowDatset(os.getcwd() + "\\data\\data.csv")
    # test = CSVDataset()
    # test.__getbatch__(3, 1)
    # Load dataset for training
    # dataset = CSVDataset()

    # Train and then test the model based on the inputted args.
    Train_Test(test)

    # Test different hyper parameter sets on the dataset
    # MassHyperparameterTest(dataset, "./param_batch.csv", "test method MassHyperparameterTest", 5, "./test_folder", "test")


if __name__ == '__main__':
    main()