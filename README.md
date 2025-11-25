# Variational autoencoder for anomaly detection

**Team members**: Carlos Muratalla-Sanchez, Cooper Cox, Joe Ewert, John Sbur <br>
**Organization**: Western Washington University, Computer Science Department<br>
**Project Desctiption**: Our goal was to create a system that is able to detect anomalies in netflow data. Our solution is the implementation of a Variational Autoencoder (VAE) which is trained on non-anomalous netflow data. This approach is based on the work of An et al. as well as an unofficial implementation of their work by Michdev.<br>
- [Variational Autoencoder based Anomaly Detection using Reconstruction Probability by Jinwon An, Sungzoon Cho](https://www.semanticscholar.org/paper/Variational-Autoencoder-based-Anomaly-Detection-An-Cho/061146b1d7938d7a8dae70e3531a00fceb3c78e8)
- [Unofficial implementation](https://github.com/Michedev/VAE_anomaly_detection/blob/master/readme.md)

# Data
For our training and testing of the model, we used data from OpenSearch provided by Western Washington that monitored netflow traffic between cities in Washington (which is private). Data was extracted and stored in a .csv file with 39 features total and each feature or encoded feature receiving its own column(s). Some of these features, like port number and protocol, have been encoded as one hot vectors and thus are represented by multiple features within the dataset. Additionally, feature values have a range of [0, 1]. Categorical features have values of only 0 or 1, representing if a particular netflow has that feature or not. Numerical data has been linearly scaled to fit between 0 and 1. The features used, and whether they are categorical or numerical, are as follows:<br>
- Destination port (categorical): The port of the device the netflow indicates traffic moved towards. Represented as a one hot vector through the following features
  - dest_port_HTTP: HTTP (port 80 or 8080)
  - dest_port_HTTPS: HTTPS (port 443)
  - dest_port_SSH: SSH (port 22 or 922)
  - dest_port_DNS: DNS (port 53)
  - dest_port_DHCP: DHCP (port 67 or 68)
  - dest_port_SMTP: SMTP (port 25)
  - dest_port_SNMP: SNMP (port 161 or port 162)
  - dest_port_RDP: RDP (port 3389)
  - dest_port_SQL: SQL (port 3306)
  - dest_port_FTP: FTP (port 20 or 21)
  - dest_port_public: any ports not encoded in range [0, 1023]
  - dest_port_private: any ports not encoded in range [1024, 49151]
  - dest_port_dynamic: any ports in range [49152, 65535]
- Source port (categorical): The port of the device the netflow originates from. Follows same encoding of ports as 'Destination port' based features. Represented as a one hot vector through the following features
  -  src_port_HTTP
  -  src_port_HTTPS
  -  src_port_SSH
  -  src_port_DNS
  -  src_port_DHCP
  -  src_port_SMTP
  -  src_port_SNMP
  -  src_port_RDP
  -  src_port_SQL
  -  src_port_FTP
  -  src_port_public
  -  src_port_private
  -  src_port_dynamic
- Protocol (categorical): The protocol of the netflow. Represented as a one hot vector through the following features
  - protocol_tcp
  - protocol_udp
  - protocol_icmp
  - protocol_ipv6-icmp
  - protocol_gre
  - protocol_esp
  - protocol_other: Encompassing any other protocols not encoded
- Bytes to client (numerical): The bytes transferred to the perceived client. Represented as a single feature named 'bytes_toclient'
- Packets to client (numerical): The packets transferred to the perceived client. Represented as a single feature named 'pkts_toclient'
- Bytes to server (numerical): The bytes transferred to the perceived server. Represented as a single feature named 'bytes_toserver'
- Packets to server (numerical): The packets transferred to the perceived server. Represented as a single feature named 'pkts_toserver'
- Direction (categorical): The direction of the netflow. Represented as a single feature named 'direction' where 0 represents the netflow being towards the client and 1 towards the server.
- Duration (numerical): The duration of the netflow in seconds. Unlike other features, this isn't linearly mapped to be in range [0,1]. Represented as a single feature named 'duration'

For each feature listed, there exists a column in the .csv file we used for training and testing our VAE. The final 40th column in the dataset we set as the 'label' column, which had range [0,1] and marked netflows as benign (0) or anomalous (1). Additionally, the .csv file's first row had headers for each feature with names as described (dest_port_HTTP, dest_port_HTTPS, ... src_port_HTTP, src_port_HTTPS, ..., protocol_tcp, ..., bytes_toclient, ...) with the final rightmost column being 'label'.<br>

While this was our schema for the data, the system is smart enough to pick up any encoded .csv datasets using the same format. As long as the last column in the .csv file is 'label', the first row is the column headers, and all values below the first row are numbers, then the VAE should train and test on that dataset just fine.<br>

## How to install

We used python 3.12.8 for this project:<br>
[Python download](https://www.python.org/downloads/)<br>

Along with this, we used pip as our library manager:<br>
[Pip installation](https://pip.pypa.io/en/stable/installation/)<br>

The libraries required for this project, along with the pip command to install them, are as follows:<br>
   
   pip install torch==2.9.0<br>
   pip install torchvision==0.24.0<br>
   pip install torchmetrics==1.8.2<br>
   pip install path==17.1.1<br>
   pip install pyyaml==6.0.3<br>
   pip install pytorch-lightning==2.5.6<br>
   pip install pandas==2.3.3<br>
   pip install scikit-learn==1.7.2<br>
   pip install requests==2.32.5<br>
   pip install tqdm==4.67.1<br>

## How to train the model

Before training the model, you'll need to bring the data from your .csv file into memory. You can do so by creating an instance of VAEDataset. Create an instance using the following constructor configuration:<br><br>

VAEDataset(
  from_file=True,
  data_file_path={path}
)<br><br>

where {path} is the path to your .csv file on your device. The method expects the .csv file to contain data in the schema specified in the 'Data' section. After creating the VAEDataset instance, you're ready to train the model.<br>

In Pipeline.py, you'll find a method named 'train()' that is responsible for training and evaluating the VAE model. It is formatted:<br><br>

   train(<br>
        args: dict, <br>
        name_of_model: str,<br>
        dataset: VAEDataset, <br>
        train_dataset_size: int = 100,<br>
        validation_dataset_size: int = 100,<br>
        test_dataset_size: int = 100,<br>
        model_class=None<br>
    ):<br><br>

  The details of each argument is as follows:<br>
  - args: dict
      - This is responsible for holding the hyperparameters used in training. The dictionary is expected to be populated with certain hyperparameter key-value pairs, detailed in the 'Hyperparameters' section. The important part is that any dictionary with all hyperparameters as described in that section will work as an argument.
  - name_of_model: str
      - Name of the specific model instance. After training, the model is stored in the 'saved models' folder. The folder within 'saved models' containing the model will have the same name as the inputted argument.
  - dataset: VAEDataset
      - The previously instantiated dataset
  - train_dataset_size: int
      - The number of benign netflow data row from 'dataset' you want to be available to use in model training
  - validation_dataset_size: int
      - The number of benign netflow data row from 'dataset' you want to be available use in model validation
  - test_dataset_size: int
      - The number of anomalous and benign netflow data row from 'dataset' you want to be available use in model testing. Uses test_dataset_size/2 anomalous netflows (rounded up) and test_dataset_size/2 benign netflows (rounded down)
  - model_class: object
      - The type of model to be trained, represented by an object name. If left blank, it defaults to 'VAEAnomalyTabular' which is the model located in VAE.py, but any model that has the same methods as 'VAEAnomalyTabular', its parent 'VAEAnomalyDetection', and inherits from nn.Module will work.
<br>
  Run this method to train the model. Afterwards, the model will be placed into the 'saved_models' folder.

## Hyperparameters

  Before training, you may want to set different hyperparameters to adjust how the model is trained. To do so, when calling Main.py or wherever execution is, you can add arguments. These arguments can be accessed by the method 'get_args()' in the form of a dictionary. For model training, there are two required hyperparameters and many optional hyperparameters. The details of each are as follows:<br>

  **Required**:<br>
  --test-name TEST_NAME<br>
                        Name of the test<br>
  --latent-size LATENT_SIZE, -l LATENT_SIZE<br>
                        Size of the latent space<br>

  **Optional**:<br>
  --num-resamples NUM_RESAMPLES, -L NUM_RESAMPLES<br>
                        Number of resamples in the latent distribution during training<br>
                        Name of the test<br>
  --latent-size LATENT_SIZE, -l LATENT_SIZE<br>
                        Size of the latent space<br>
  --num-resamples NUM_RESAMPLES, -L NUM_RESAMPLES<br>
                        Number of resamples in the latent distribution during training<br>
                        Size of the latent space<br>
  --num-resamples NUM_RESAMPLES, -L NUM_RESAMPLES<br>
                        Number of resamples in the latent distribution during training<br>
                        Number of resamples in the latent distribution during training<br>
  --epochs EPOCHS, -e EPOCHS<br>
                        Number of epochs to train for<br>
  --batch-size BATCH_SIZE, -b BATCH_SIZE<br>
  --device {cpu,gpu,tpu}, -d {cpu,gpu,tpu}, --accelerator {cpu,gpu,tpu}<br>
                        Device to use for training. Can be cpu, gpu or tpu<br>
  --lr LR               Learning rate<br>
  --num-tests NUM_TESTS<br>
                        Number of tests run during model testing<br>
  --alpha ALPHA         Threshold for VAE when determining whether a record is an anomaly or   <br> 
                        not<br>
  --no-progress-bar<br>
  --steps-log-loss STEPS_LOG_LOSS<br>
                        Number of steps between each loss logging<br>
  --steps-log-norm-params STEPS_LOG_NORM_PARAMS<br>
                        Number of steps between each model parameters logging<br>

  Should any optional values be left blank, they will be set to their default values. These values are located in Pipeline.py under the get_args() method within the calls to 'parser.add_argument()' as metod argument 'default'.<br>

  Hyperparameters are called via the console in the format of:<br>
  {python} {.py file} {arg 1 name} {arg 1 value} {arg 2 name} {arg 2 value}   // etc.<br>

  Where:<br>
    {python}: Path to python executable on your device<br>
    {.py file}: File to execute<br>
    {arg 1 name}: Name of argument with two '-' beforehand as seen in the hyperparameter descriptions.<br>
    {arg 1 value}: Value of arg 1<br>

  For example:<br>
    ./python.exe ./Main.py --test-name 'First test' --latent-size 6 --epochs 10<br>


## How to evaluate model performance:

The method 'train_test()' has been set up to train and test a VAEAnomalyTabular model instance based on an inputted VAEDataset and the arguments inputted through the console as specified by the 'Hyperparameters' section. Running the method with an inputted VAEDataset will output a model in 'saved models' with the name specified by the hyperparameter 'test-name' as well as test results in the form of a .csv file with the same model name under 'test folder'.

## How to make predictions:
Once the model is trained, load and predict with this code snippet:<br>

import torch<br>

#load X_test<br>
#(Insert code to load X_test here)<br>
#(X_test expected to be a Tensor with number of rows equal to 'batch_size' and number of columns equal to the number of features)<br><br>

#load model<br>
model = ModelInfo(model_id={name}, model_input_size={input size}) <br>
#replace 'name' with the name of the model as it appears in the 'saved models' folder and replace 'input size' with the number of features used to train that model.<br><br>

#test model<br>
outliers = model.is_anomaly(X_test)<br>

