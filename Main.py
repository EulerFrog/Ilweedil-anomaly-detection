
import ModelInfo
from Download import download_netflow_dataset, stat_netflow_dataset
from Dataset import VAEDataset
from Pipeline import train_test
from ModelInfo import ModelInfo
import os

def main():

    # Load dataset 
    dataset = VAEDataset(from_file=True, data_file_path=os.getcwd() + "/data/data.csv")

    # Train and test model on dataset
    train_test(
        dataset=dataset,
        train_benign_dataset_size_percentage=0.001,
        train_anomaly_dataset_size_percentage=0.2,
        test_benign_dataset_size_percentage=0.05,
        test_anomaly_dataset_size_percentage=0.6
        )

    # model = ModelInfo(
    #     "test joe integration",
    #     39
    # )
    # model.LoadModel()

    # ls_start = 10
    # ls_end = 30 # 5
    # ls_step = 4
    # bs_start = 20 # 3
    # bs_end = 100
    # bs_step = 40
    # nr_start = 10 # 1
    # nr_end = 10
    # nr_step = 0
    # e_start = 5 # 5
    # e_end = 15
    # e_step = 2
    # a_start = 1 # 8
    # a_end = 9
    # a_step = 1

    # with open("batches.txt", "w") as f:

    #     for ls in range(ls_start, ls_end+1, ls_step):
    #         for bs in range(bs_start, bs_end+1, bs_step):
    #             for e in range(e_start, e_end+1, e_step):
    #                 for a in range(a_start, a_end+1, a_step):
    #                     a = a / 10
    #                     write_string = "ls_"+str(ls)+"_bs_"+str(bs)+"_nr_"+str(nr_start)+"_e_"+str(e)+"_lr_0.001_a_"+str(a)+"_nt_30"
    #                     write_string += " "
    #                     write_string += str(ls)
    #                     write_string += " "
    #                     write_string += str(bs)
    #                     write_string += " "
    #                     write_string += str(nr_start)
    #                     write_string += " "
    #                     write_string += str(e)
    #                     write_string += " "
    #                     write_string += str(0.001)
    #                     write_string += " "
    #                     write_string += str(a)
    #                     write_string += " 30\n"
    #                     f.write(
    #                         write_string
    #                     )

    # model.MassTestModel(
    #     dataset,
    #     1000,
    #     1000,
    #     100,
    #     os.getcwd() + '\\test_folder' + "\\(##)" + "test_name",
    #     "verify results",
    #     -200  
    # )

if __name__ == '__main__':
    main()
