import argparse
from datetime import datetime
import os

from ModelInfo import ModelInfo
from Dataset import VAEDataset
import torch
import yaml
from path import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import math

from model.VAE import VAEAnomalyTabular
from torch.utils.data import Dataset, TensorDataset

ROOT = Path(__file__).parent
SAVED_MODELS = ROOT / 'saved_models'

def make_folder_run(name=None) -> Path:
    """
    Get the folder where to store the experiment.
    The folder is named with the current date and time.

    Returns:
        Path: the path to the folder where to store the experiment 
    """
    
    folder_name = ""
    if (name is None):
        folder_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    else:
        folder_name = name
    checkpoint_folder = SAVED_MODELS / folder_name
    checkpoint_folder.makedirs_p()
    return checkpoint_folder

def get_args() -> argparse.Namespace:
    """
    Parse command line arguments

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test-name', 
        type=str,
        required=True,
        dest='test_name',
        help='Name of the test'
    )
    parser.add_argument(
        '--latent-size', '-l',
        type=int,
        required=True,
        dest='latent_size',
        help='Size of the latent space'
    )
    parser.add_argument(
        '--num-resamples', '-L',
        type=int,
        dest='num_resamples',
        default=10,
        help='Number of resamples in the latent distribution during training'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        dest='epochs',
        default=100,
        help='Number of epochs to train for'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        dest='batch_size',
        default=32
    )
    parser.add_argument(
        '--device', '-d', '--accelerator',
        type=str,
        dest='device',
        default='cpu',
        help='Device to use for training. Can be cpu, gpu or tpu',
        choices=['cpu', 'gpu', 'tpu']
    )
    parser.add_argument(
        '--lr',
        type=float,
        dest='lr',
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--num-tests',
        type=int,
        dest='num_tests',
        default=5,
        help='Number of tests run during model testing'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        dest='alpha',
        default=0.5,
        help='Threshold for VAE when determining whether a record is an anomaly or not'
    )
    parser.add_argument(
        '--no-progress-bar',
        action='store_true',
        dest='no_progress_bar'
    )
    parser.add_argument(
        '--steps-log-loss',
        type=int,
        dest='steps_log_loss',
        default=1_000,
        help='Number of steps between each loss logging'
    )
    parser.add_argument(
        '--steps-log-norm-params',
        type=int,
        dest='steps_log_norm_params',
        default=1_000,
        help='Number of steps between each model parameters logging'
    )
    return parser.parse_args()

def train(
        args: dict, 
        name_of_model: str,
        dataset: VAEDataset, 
        train_dataset_size: int = 100,
        validation_dataset_size: int = 100,
        test_dataset_size: int = 100,
        model_class=None
    ):
    """
    Main function to train the VAE model

    Args:
        args: dict - Command-line arguments in dictionary form.
        name_of_model: str - Name of the instance of the model trained, evaluated, and tested
        dataset: VAEDataset - dataset containing data loaders for the pipeline.
        train_dataset_size: int - represents the number of benign records the dataloader used in training should have
        validation_dataset_size: int - represents the number of benign records the dataloader used in validation should have
        test_dataset_size: int - represents the number of anomalou records the dataloader used in validation should have
        model_class: object - The autoencoder class to use (e.g., VAEAnomalyTabular). If None, uses default.
    """
    # Locals
    experiment_folder = None
    checkpoint_folder = None
    model = None
    train_dloader = None
    val_dloader = None
    test_dloader = None

    print(args)

    experiment_folder = make_folder_run(name_of_model)

    with open(experiment_folder / 'config.yaml', 'w') as f:
        yaml.dump(args, f)

    checkpoint_folder = experiment_folder / 'checkpoints'
    checkpoint_folder.makedirs_p()

    # Setup device
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
                              
    # Use default model class if not provided
    if model_class is None:
        model_class = VAEAnomalyTabular

    # Initialize model
    model = model_class(
        dataset.input_size,
        args.latent_size,
        args.num_resamples,
        lr=args.lr
    )
    model = model.to(device)

    # Retrieve data loaders from VAE dataset
    train_dloader = dataset.get_benign_dataloader(
        batch_size=args.batch_size, 
        size = train_dataset_size
        )
    val_dloader = dataset.get_benign_dataloader(
        batch_size=args.batch_size, 
        size = validation_dataset_size
        )
    test_dloader = dataset.get_anomalous_dataloader(
        batch_size=args.batch_size,
        size=test_dataset_size
    )

    # Create checkpoint folder
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_folder,
        filename='{epoch:02d} {val_loss_epoch:.2f}',
        save_top_k=-1,
        verbose=True,
        monitor='val_loss_epoch',
        mode='min',
        save_last=True,
        every_n_epochs=1,
    )

    # Train model
    # trainer = Trainer(
    #     callbacks=[checkpoint],
    #     max_epochs=args.epochs,
    #     accelerator=args.device,
    #     enable_progress_bar=not args.no_progress_bar,
    # )
    # trainer.fit(model, train_dloader, val_dloader)


    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Setup TensorBoard writer
    writer = SummaryWriter(log_dir=str(experiment_folder / 'logs'))

    # Training loop
    best_val_loss = float('inf')
    global_step = 0

    f = open("./measuring_loss_over_time.txt", "w")

    for epoch in range(args.epochs):

        #
        #  --- Training phase ---
        #
        model.train()
        train_loss_total = 0
        train_kl_total = 0
        train_recon_total = 0

        train_iterator = tqdm(train_dloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]",
                            disable=args.no_progress_bar)

        for batch_idx, batch in enumerate(train_iterator):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model.forward(x) 
            loss = output['loss']

            # Backward pass
            loss.backward()
            optimizer.step()
            # Accumulate losses
            train_loss_total += loss.item()
            train_kl_total += output['kl'].item()
            train_recon_total += output['recon_loss'].item()

            # Log to TensorBoard at specified intervals
            if global_step % args.steps_log_loss == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/kl_divergence', output['kl'].item(), global_step)
                writer.add_scalar('train/recon_loss', output['recon_loss'].item(), global_step)

            if global_step % args.steps_log_norm_params == 0:
                # Log parameter norms
                param_norm = sum(p.norm(1).item() for p in model.parameters())
                grad_norm = sum(p.grad.norm(1).item() for p in model.parameters() if p.grad is not None)
                writer.add_scalar('train/param_norm', param_norm, global_step)
                writer.add_scalar('train/grad_norm', grad_norm, global_step)

            global_step += 1

            # Update progress bar
            if not args.no_progress_bar:
                train_iterator.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'kl': f'{output["kl"].item():.4f}',
                    'recon': f'{output["recon_loss"].item():.4f}'
                })

        # Calculate average training losses
        avg_train_loss = train_loss_total / len(train_dloader)
        avg_train_kl = train_kl_total / len(train_dloader)
        avg_train_recon = train_recon_total / len(train_dloader)

        #
        #  --- End training phase ---
        #

        #
        #  --- Validation phase ---
        #
        model.eval()
        val_loss_total = 0
        val_kl_total = 0
        val_recon_total = 0

        val_iterator = tqdm(val_dloader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]",
                          disable=args.no_progress_bar)

        with torch.no_grad():
            for batch in val_iterator:
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)

                # Forward pass
                output = model.forward(x)

                # Accumulate losses
                val_loss_total += output['loss'].item()
                val_kl_total += output['kl'].item()
                val_recon_total += output['recon_loss'].item()
                     

        # Calculate average validation losses
        avg_val_loss = val_loss_total / len(val_dloader)
        avg_val_kl = val_kl_total / len(val_dloader)
        avg_val_recon = val_recon_total / len(val_dloader)

        #
        #  --- End validation phase ---
        #

        #
        #  --- Testing phase ---
        #

        # Anomaly detection evaluation
        anomaly_recon_errors = []
        anomaly_labels = []

        #   Test iterator to contain both anomalous and benign data
        test_iterator = tqdm(test_dloader, desc=f"Epoch {epoch+1}/{args.epochs} [Anomaly Eval]",
                               disable=args.no_progress_bar)

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_iterator):
                # Handle different batch formats
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    # Batch contains (data, labels)
                    x, batch_labels = batch
                    x = x.to(device)
                    batch_labels_list = batch_labels.cpu().numpy().tolist()
                elif isinstance(batch, (list, tuple)) and len(batch) == 1:
                    # Batch contains only data (wrapped in list/tuple)
                    x = batch[0].to(device)
                    batch_labels_list = None
                else:
                    # Batch contains only data
                    x = batch.to(device)
                    batch_labels_list = None

                # Calculate per-sample reconstruction errors
                # Get the reconstruction distribution 

                pred_result = model.predict(x)
                x_expanded = x.unsqueeze(0)  # Shape: [1, batch_size, features]
                recon_dist = torch.distributions.Normal(pred_result['recon_mu'], pred_result['recon_sigma'])

                # Calculate negative log-likelihood per sample (reconstruction error)
                # log_prob: [L, batch_size, features] -> mean over L and sum over features
                log_lik_per_sample = recon_dist.log_prob(x_expanded).mean(dim=0).sum(dim=-1)  # [batch_size]
                recon_error_per_sample = -log_lik_per_sample  # Negative log-likelihood as error

                # Store reconstruction errors
                batch_recon_errors = recon_error_per_sample.detach().cpu().numpy()
                anomaly_recon_errors.extend(batch_recon_errors)

                # Get labels for this batch
                if batch_labels_list is not None:
                    # Labels are in the batch
                    anomaly_labels.extend(batch_labels_list)
                else:
                    # Fallback: get labels from dataset if available
                    start_idx = batch_idx * args.batch_size
                    end_idx = min(start_idx + args.batch_size, len(test_iterator))
                    for idx in range(start_idx, end_idx):
                        label = test_iterator.__getitemlabel__(idx)
                        # Binary label: 1 if anomaly (class n-1), 0 if normal (classes 0 to n-2)
                        is_anomaly = 1 if label == test_iterator.anomaly_class else 0
                        anomaly_labels.append(is_anomaly)

        # Convert to numpy arrays
        anomaly_recon_errors = np.array(anomaly_recon_errors)
        anomaly_labels = np.array(anomaly_labels)

        # Calculate threshold based on validation set
        # avg_val_recon is log-likelihood (positive for normal samples)
        # We're using -log_lik as reconstruction error, so negate and scale
        # avg_val_recon_error = -avg_val_recon  # Convert to reconstruction error
        # threshold = avg_val_recon_error * 1.5  # Higher error = more likely anomaly

        avg_val_recon_error = -avg_val_recon
        threshold = avg_val_recon_error * 1.5

        # print("Avg val recon error: ")
        # print(avg_val_recon_error)
        # print("Anomaly Threshold: ")
        # print(threshold)

        # i = 0
        # for i in range(0,100):
        #     print("Test1: Threshold multiplier <" + str(i*0.01) + "> Threshold <" + str(avg_val_recon_error * (1+(i*0.01))) + "> Avg val recon <" + str(avg_val_recon_error) + ">")
        #     predictions = (anomaly_recon_errors < avg_val_recon_error * (1+(i*0.01))).astype(int)
        #     true_positives = np.sum((predictions == 1) & (anomaly_labels == 1))
        #     false_positives = np.sum((predictions == 1) & (anomaly_labels == 0))
        #     false_negatives = np.sum((predictions == 0) & (anomaly_labels == 1))
        #     true_negatives = np.sum((predictions == 0) & (anomaly_labels == 0))
        #     print(f"Total: {predictions.__len__()} - TP: <{true_positives}> FP: <{false_positives}> FN: <{false_negatives}> TN: <{true_negatives}>")

        # input()

        # Make predictions: 1 if reconstruction error > threshold, 0 otherwise
        predictions = (anomaly_recon_errors > (threshold)).astype(int)

        # Calculate confusion matrix elements
        true_positives = np.sum((predictions == 1) & (anomaly_labels == 1))
        false_positives = np.sum((predictions == 1) & (anomaly_labels == 0))
        false_negatives = np.sum((predictions == 0) & (anomaly_labels == 1))
        true_negatives = np.sum((predictions == 0) & (anomaly_labels == 0))

        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Calculate average reconstruction error for normal vs anomaly samples
        # normal_mask = anomaly_labels == 0
        # anomaly_mask = anomaly_labels == 1
        # avg_normal_recon = np.mean(anomaly_recon_errors[normal_mask]) if np.any(normal_mask) else 0.0
        # avg_anomaly_recon = np.mean(anomaly_recon_errors[anomaly_mask]) if np.any(anomaly_mask) else 0.0

        # Log epoch metrics to TensorBoard
        writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)
        writer.add_scalar('epoch/train_kl', avg_train_kl, epoch)
        writer.add_scalar('epoch/train_recon', avg_train_recon, epoch)
        writer.add_scalar('epoch/val_loss', avg_val_loss, epoch)
        writer.add_scalar('epoch/val_kl', avg_val_kl, epoch)
        writer.add_scalar('epoch/val_recon', avg_val_recon, epoch)
        writer.add_scalar('epoch/anomaly_precision', precision, epoch)
        writer.add_scalar('epoch/anomaly_recall', recall, epoch)
        writer.add_scalar('epoch/anomaly_f1', f1_score, epoch)
        writer.add_scalar('epoch/anomaly_threshold', threshold, epoch)
        # writer.add_scalar('epoch/normal_recon_error', avg_normal_recon, epoch)
        # writer.add_scalar('epoch/anomaly_recon_error', avg_anomaly_recon, epoch)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Train KL: {avg_train_kl:.4f} | Val KL: {avg_val_kl:.4f}")
        print(f"  Train Recon: {avg_train_recon:.4f} | Val Recon: {avg_val_recon:.4f}")
        print(f"\n  Anomaly Detection (threshold={threshold:.4f}):")
        print(f"    Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1_score:.4f}")
        # print(f"    Normal Recon: {avg_normal_recon:.4f} | Anomaly Recon: {avg_anomaly_recon:.4f}")
        print(f"    TP: {true_positives} | FP: {false_positives} | FN: {false_negatives} | TN: {true_negatives}")


        f.write(f"\nEpoch {epoch+1}/{args.epochs} Summary:\n")
        f.write(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")
        f.write(f"  Train KL: {avg_train_kl:.4f} | Val KL: {avg_val_kl:.4f}")
        f.write(f"  Train Recon: {avg_train_recon:.4f} | Val Recon: {avg_val_recon:.4f}\n")
        f.write(f"\n  Anomaly Detection (threshold={threshold:.4f}):")
        f.write(f"    Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1_score:.4f}\n")
        f.write(f"    TP: {true_positives} | FP: {false_positives} | FN: {false_negatives} | TN: {true_negatives}\n")
        f.write(f"**************************************************************************************************")
        

        # Save checkpoint for this epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'config': args
        }

        torch.save(checkpoint, checkpoint_folder / f'epoch_{epoch:03d}_val_loss_{avg_val_loss:.4f}.ckpt')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, checkpoint_folder / 'best.ckpt')
            print(f"  New best model saved with val_loss: {best_val_loss:.4f}")

        # Save last checkpoint
        torch.save(checkpoint, checkpoint_folder / 'last.ckpt')

    f.close()

    print("\nTraining completed!")
    print(f"Model saved in: {experiment_folder}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    writer.close()

    # When finished, return path to completed model at last checkpoint
    return 

def train_test(
        dataset: VAEDataset,
        train_benign_dataset_size_percentage: float,
        train_anomaly_dataset_size_percentage: float,
        test_benign_dataset_size_percentage: float,
        test_anomaly_dataset_size_percentage: float,
    ):
    """
        Trains and then tests a model based on args inputted to python. 

        Outputs the test to the "tests_folder" in the cwd.

        *In the case that tests share the same name, an increasing counter is added to the name of the test:
        e.g. (for test name of "test")
            test
            test 1
            test 2
            .
            .
            test n

        Args:
            dataset: VAEDataset - VAE dataset to train/test on
            train_benign_dataset_size_percentage: float - For training, the percentage of all benign data to use for the training/validation (and of the percentage, train will be allocated 80% and validate 20%)
            train_anomaly_dataset_size_percentage: float -  For training, the percentage of all anomalous data to use for the testing 
            test_benign_dataset_size_percentage: float - For test, the percentage of all benign data to use for the test set
            test_anomaly_dataset_size_percentage: float - For test, the percentage of all anomalous data to use for the test set

            Note: 
                (train_benign_dataset_size_percentage + test_benign_dataset_size_percentage) should add up to 1 or less
                (train_anomaly_dataset_size_percentage + test_anomaly_dataset_size_percentage) should add up to 1 or less
    """

    # Locals
    args = None
    test_model_path = ""
    test_folder_name = os.getcwd() + '\\test_folder'
    str_holder = ""
    i = 1
    train_train_benign_size = 0.0
    train_validate_benign_size = 0.0
    train_test_anomalous_size = 0.0
    test_anomalous_size = 0.0
    test_benign_size = 0.0
    temp_float1 = 0
    temp_float2 = 0

    # Get arguments
    args = get_args()

    # Create "test_folder" if it doesn't exist already
    if not os.path.isdir(test_folder_name):
        os.mkdir(test_folder_name)

    # Within test_folder, check for models of the same name as this test. If they exist, create a new
    #   name for the model that isn't seen in the folder.
    test_model_path = test_folder_name + "\\(##)" + args.test_name
    str_holder = test_model_path
    while os.path.isdir(str_holder):
        str_holder = test_model_path + " " + str(i)
        i = i + 1
    test_model_path = str_holder    
    if (i > 1):
        args.test_name = args.test_name + " " + str(i)


    # Calculate size of data loaders for train/validate/test
    #   Validate split arguments are valid
    temp_float1 = train_benign_dataset_size_percentage + test_benign_dataset_size_percentage
    temp_float2 = train_anomaly_dataset_size_percentage + test_anomaly_dataset_size_percentage
    if  (temp_float1 > 1 
         or temp_float2 > 1 
         or (train_benign_dataset_size_percentage > 1 or train_benign_dataset_size_percentage < 0)
         or (test_benign_dataset_size_percentage > 1 or test_benign_dataset_size_percentage < 0)
         or (train_anomaly_dataset_size_percentage > 1 or train_anomaly_dataset_size_percentage < 0)
         or (test_anomaly_dataset_size_percentage > 1 or test_anomaly_dataset_size_percentage < 0)):
        err_str = f'train_test() (err): Invalid split entered (total benign allocation requested = '
        err_str = err_str + f"{temp_float1} and total anomalous allocation requested = {temp_float2}; requested percentages -  )"
        err_str = err_str + f" train_benign_dataset_size_percentage: {train_benign_dataset_size_percentage}; "
        err_str = err_str + f" test_benign_dataset_size_percentage: {test_benign_dataset_size_percentage}; "
        err_str = err_str + f" train_anomaly_dataset_size_percentage: {train_anomaly_dataset_size_percentage}; "
        err_str = err_str + f" test_anomaly_dataset_size_percentage: {test_anomaly_dataset_size_percentage};) "
        raise Exception(err_str)
    
    #   Create sizes of each partition
    train_train_benign_size = math.floor(dataset.benign_data_length * train_benign_dataset_size_percentage * 0.8)
    train_validate_benign_size = math.floor(dataset.benign_data_length * train_benign_dataset_size_percentage * 0.2)
    train_test_anomalous_size = math.floor(dataset.anomalous_data_length * train_anomaly_dataset_size_percentage)
    test_benign_size = math.floor(dataset.benign_data_length * test_benign_dataset_size_percentage)
    test_anomalous_size = math.floor(dataset.anomalous_data_length * test_anomaly_dataset_size_percentage)

    # Train the model
    train(
        args, 
        args.test_name, 
        dataset,
        test_dataset_size=train_test_anomalous_size,
        train_dataset_size=train_train_benign_size,
        validation_dataset_size=train_validate_benign_size,
        model_class=VAEAnomalyTabular
    )

    # Test the model
    model = ModelInfo(args.test_name, dataset.input_size)
    model.LoadModel()
    model.MassTestModel(
        dataset, 
        test_anomalous_size,
        test_benign_size,
        args.num_tests, 
        test_model_path, 
        args.test_name
        )
