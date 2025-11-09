
import ModelInfo
from dataset import CSVDataset
from methods import MassHyperparameterTest

def main():
    

    # Load dataset for training
    dataset = CSVDataset()

    # Test different hyper parameter sets on the dataset
    MassHyperparameterTest(dataset, "./param_batch.csv", "test method MassHyperparameterTest", 5, "./test_folder", "test")


if __name__ == '__main__':
    main()