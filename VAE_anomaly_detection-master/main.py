
import ModelInfo
from dataset import CSVDataset
from methods import Train_Test

def main():

    # Load dataset for training
    dataset = CSVDataset()

    # Train and then test the model based on the inputted args.
    Train_Test(dataset)

    # Test different hyper parameter sets on the dataset
    # MassHyperparameterTest(dataset, "./param_batch.csv", "test method MassHyperparameterTest", 5, "./test_folder", "test")


if __name__ == '__main__':
    main()