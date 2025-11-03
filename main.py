import argparse
from datetime import datetime
import json
from pathlib import Path

from utility import ModelInfo

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import mnist_dataset, test_dataset, SyntheticAnomalyDataset
from model.VAE import VAEAnomalyTabular
from torch.utils.data import Dataset, TensorDataset

ROOT = Path(__file__).parent
SAVED_MODELS = ROOT / 'saved_models'


def make_folder_run() -> Path:
    """
    Get the folder where to store the experiment.
    The folder is named with the current date and time.

    Returns:
        Path: the path to the folder where to store the experiment 
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_folder = SAVED_MODELS / timestamp
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    return checkpoint_folder


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-size', '-i',
        type=int,
        required=True,
        dest='input_size',
        help='Number of input features. In 1D case it is the vector length, in 2D case it is the number of channels'
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
        default='gpu',
        help='Device to use for training. Can be cpu, gpu or tpu',
        choices=['cpu', 'gpu']
    )
    parser.add_argument(
        '--lr',
        type=float,
        dest='lr',
        default=1e-3,
        help='Learning rate'
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


def train():
    """
    Main function to train the VAE model
    """
    args = get_args()
    print(args)

    experiment_folder = make_folder_run()

    # Save config as JSON instead of YAML
    config_dict = vars(args)
    with open(experiment_folder / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)

    checkpoint_folder = experiment_folder / 'checkpoints'
    checkpoint_folder.mkdir(parents=True, exist_ok=True)

    # Setup device
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Initialize model
    model = VAEAnomalyTabular(
        args.input_size,
        args.latent_size,
        args.num_resamples,
        lr=args.lr
    )
    model = model.to(device)

    # Setup datasets with three-way split
    print("\n=== Loading Datasets ===")

    # Dataset parameters - 9 classes total (0-7 normal, 8 anomaly)
    n_classes = 9
    dataset_params = {
        'n_samples': 10000,
        'n_features': args.input_size,
        'n_classes': n_classes,
        'n_informative': max(0, int(args.input_size * 0.5)),  # 80% informative features
        'n_redundant': max(0, int(args.input_size * 0.5)),     # 10% redundant features
        'anomaly_ratio': 0.02,
        'random_state': 42,
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'test_ratio': 0.2,
        'class_sep': 2.5  # Increased separation for more distinct classes
    }

    train_set = SyntheticAnomalyDataset(**dataset_params, split='train')
    train_dloader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=0)

    val_dataset = SyntheticAnomalyDataset(**dataset_params, split='val')
    val_dloader = DataLoader(val_dataset, args.batch_size, num_workers=0)

    # Anomaly set for evaluation during training (contains both normal and anomalous samples)
    anomaly_dataset = SyntheticAnomalyDataset(**dataset_params, split='test')
    anomaly_dloader = DataLoader(anomaly_dataset, args.batch_size, num_workers=0)

    print(f"\nNote: Training and validation use only normal samples (classes 0-{n_classes-2})")
    print(f"      Class {n_classes-1} is designated as the anomaly class")
    print(f"      Anomaly set contains both normal and anomalous samples for evaluation\n")

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Setup TensorBoard writer
    writer = SummaryWriter(log_dir=str(experiment_folder / 'logs'))

    # Training loop
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(args.epochs):
        # Training phase
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

        # Validation phase
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

        # Anomaly detection evaluation
        anomaly_recon_errors = []
        anomaly_labels = []

        anomaly_iterator = tqdm(anomaly_dloader, desc=f"Epoch {epoch+1}/{args.epochs} [Anomaly Eval]",
                               disable=args.no_progress_bar)

        with torch.no_grad():
            for batch_idx, batch in enumerate(anomaly_iterator):
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)

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
                start_idx = batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(anomaly_dataset))
                for idx in range(start_idx, end_idx):
                    label = anomaly_dataset.__getitemlabel__(idx)
                    # Binary label: 1 if anomaly (class n-1), 0 if normal (classes 0 to n-2)
                    is_anomaly = 1 if label == anomaly_dataset.anomaly_class else 0
                    anomaly_labels.append(is_anomaly)

        # Convert to numpy arrays
        anomaly_recon_errors = np.array(anomaly_recon_errors)
        anomaly_labels = np.array(anomaly_labels)

        # Calculate threshold based on validation set
        # avg_val_recon is log-likelihood (positive for normal samples)
        # We're using -log_lik as reconstruction error, so negate and scale
        avg_val_recon_error = -avg_val_recon  # Convert to reconstruction error
        threshold = avg_val_recon_error * 1.5  # Higher error = more likely anomaly

        # Make predictions: 1 if reconstruction error > threshold, 0 otherwise
        predictions = (anomaly_recon_errors > threshold).astype(int)

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
        normal_mask = anomaly_labels == 0
        anomaly_mask = anomaly_labels == 1
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

        # Save checkpoint for this epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'config': config_dict
        }

        torch.save(checkpoint, checkpoint_folder / f'epoch_{epoch:03d}_val_loss_{avg_val_loss:.4f}.ckpt')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, checkpoint_folder / 'best.ckpt')
            print(f"  New best model saved with val_loss: {best_val_loss:.4f}")

        # Save last checkpoint
        torch.save(checkpoint, checkpoint_folder / 'last.ckpt')

    print("\nTraining completed!")
    print(f"Model saved in: {experiment_folder}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    writer.close()
    return experiment_folder 


def main():
    
    # test = ModelInfo("2025-10-22_17-35-14")
    # test.LoadModel()
    # test.TestModel(test_dataset())
    train()


if __name__ == '__main__':
    main()