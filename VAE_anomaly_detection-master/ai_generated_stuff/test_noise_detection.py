import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import mnist_dataset
from model.VAE import VAEAnomalyTabular


def generate_random_noise(num_samples, input_size):
    """
    Generate random noise images.

    Args:
        num_samples: Number of noise samples to generate
        input_size: Size of each sample (784 for MNIST)

    Returns:
        Tensor of random noise
    """
    # Generate uniform random noise in [0, 1] range (like MNIST normalization)
    noise = torch.rand(num_samples, input_size)
    return noise


def generate_gaussian_noise(num_samples, input_size):
    """
    Generate Gaussian noise images.

    Args:
        num_samples: Number of noise samples to generate
        input_size: Size of each sample (784 for MNIST)

    Returns:
        Tensor of Gaussian noise
    """
    # Generate Gaussian noise and clip to [0, 1]
    noise = torch.randn(num_samples, input_size) * 0.3 + 0.5
    noise = torch.clamp(noise, 0, 1)
    return noise


def test_anomaly_detection(model, normal_data, anomaly_data, alpha=0.05):
    """
    Test anomaly detection on normal and anomalous data.

    Args:
        model: Trained VAE model
        normal_data: Normal MNIST digits
        anomaly_data: Anomalous data (noise)
        alpha: Threshold for anomaly detection

    Returns:
        Dictionary with detection statistics
    """
    model.eval()

    # Get reconstruction probabilities
    with torch.no_grad():
        normal_probs = model.reconstructed_probability(normal_data)
        anomaly_probs = model.reconstructed_probability(anomaly_data)

    # Detect anomalies
    normal_anomalies = model.is_anomaly(normal_data, alpha=alpha)
    anomaly_anomalies = model.is_anomaly(anomaly_data, alpha=alpha)

    # Calculate statistics
    stats = {
        'normal_mean_prob': normal_probs.mean().item(),
        'normal_std_prob': normal_probs.std().item(),
        'anomaly_mean_prob': anomaly_probs.mean().item(),
        'anomaly_std_prob': anomaly_probs.std().item(),
        'normal_flagged_as_anomaly': normal_anomalies.sum().item() / len(normal_data),
        'anomaly_flagged_as_anomaly': anomaly_anomalies.sum().item() / len(anomaly_data),
        'normal_probs': normal_probs.cpu().numpy(),
        'anomaly_probs': anomaly_probs.cpu().numpy()
    }

    return stats


def visualize_results(model, normal_samples, noise_samples, num_samples=5):
    """
    Visualize original images and their reconstructions.

    Args:
        model: Trained VAE model
        normal_samples: Normal MNIST samples
        noise_samples: Noise samples
        num_samples: Number of samples to visualize
    """
    model.eval()

    with torch.no_grad():
        # Get reconstructions
        normal_pred = model.predict(normal_samples[:num_samples])
        noise_pred = model.predict(noise_samples[:num_samples])

        normal_recon = normal_pred['recon_mu'].mean(dim=0)  # Average over L samples
        noise_recon = noise_pred['recon_mu'].mean(dim=0)

    # Create figure
    fig, axes = plt.subplots(4, num_samples, figsize=(15, 8))
    fig.suptitle('VAE Anomaly Detection: Normal MNIST vs Random Noise', fontsize=16)

    for i in range(num_samples):
        # Normal MNIST - Original
        axes[0, i].imshow(normal_samples[i].cpu().reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Normal\nOriginal', fontsize=10)

        # Normal MNIST - Reconstruction
        axes[1, i].imshow(normal_recon[i].cpu().reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Normal\nReconstructed', fontsize=10)

        # Random Noise - Original
        axes[2, i].imshow(noise_samples[i].cpu().reshape(28, 28), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Noise\nOriginal', fontsize=10)

        # Random Noise - Reconstruction
        axes[3, i].imshow(noise_recon[i].cpu().reshape(28, 28), cmap='gray')
        axes[3, i].axis('off')
        if i == 0:
            axes[3, i].set_title('Noise\nReconstructed', fontsize=10)

    plt.tight_layout()
    plt.savefig('noise_detection_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to 'noise_detection_visualization.png'")
    plt.close()


def plot_probability_distributions(stats):
    """
    Plot probability distributions for normal and anomalous data.

    Args:
        stats: Dictionary with detection statistics
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of probabilities
    axes[0].hist(stats['normal_probs'], bins=50, alpha=0.7, label='Normal MNIST', color='blue')
    axes[0].hist(stats['anomaly_probs'], bins=50, alpha=0.7, label='Random Noise', color='red')
    axes[0].set_xlabel('Reconstruction Probability')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Reconstruction Probabilities')
    axes[0].legend()
    axes[0].set_yscale('log')

    # Box plot
    data_to_plot = [stats['normal_probs'], stats['anomaly_probs']]
    axes[1].boxplot(data_to_plot, labels=['Normal MNIST', 'Random Noise'])
    axes[1].set_ylabel('Reconstruction Probability')
    axes[1].set_title('Reconstruction Probability Comparison')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('probability_distributions.png', dpi=150, bbox_inches='tight')
    print("Probability distributions saved to 'probability_distributions.png'")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test VAE anomaly detection with random noise')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--num-noise-samples',
        type=int,
        default=1000,
        help='Number of random noise samples to generate'
    )
    parser.add_argument(
        '--num-normal-samples',
        type=int,
        default=1000,
        help='Number of normal MNIST samples to test'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Threshold for anomaly detection'
    )
    parser.add_argument(
        '--noise-type',
        type=str,
        default='uniform',
        choices=['uniform', 'gaussian'],
        help='Type of noise to generate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for inference'
    )

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load trained model
    print(f"\nLoading model from {args.checkpoint}...")

    # MNIST is 28x28 = 784 features
    input_size = 784
    latent_size = 32  # From training
    num_resamples = 10  # From training default

    # Create model instance
    model = VAEAnomalyTabular(input_size, latent_size, L=num_resamples)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  Input size: {model.input_size}")
    print(f"  Latent size: {model.latent_size}")
    print(f"  L (resamples): {model.L}")

    # Load normal MNIST test data
    print(f"\nLoading {args.num_normal_samples} normal MNIST samples...")
    test_dataset = mnist_dataset(train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.num_normal_samples, shuffle=True)
    normal_data = next(iter(test_loader)).to(device)
    normal_data = normal_data[:args.num_normal_samples]

    # Generate random noise
    print(f"Generating {args.num_noise_samples} {args.noise_type} noise samples...")
    if args.noise_type == 'uniform':
        noise_data = generate_random_noise(args.num_noise_samples, model.input_size).to(device)
    else:
        noise_data = generate_gaussian_noise(args.num_noise_samples, model.input_size).to(device)

    # Test anomaly detection
    print(f"\nTesting anomaly detection (alpha={args.alpha})...")
    stats = test_anomaly_detection(model, normal_data, noise_data, alpha=args.alpha)

    # Print results
    print("\n" + "="*70)
    print("ANOMALY DETECTION RESULTS")
    print("="*70)
    print(f"\nNormal MNIST Digits:")
    print(f"  Mean reconstruction probability: {stats['normal_mean_prob']:.6f}")
    print(f"  Std reconstruction probability:  {stats['normal_std_prob']:.6f}")
    print(f"  Flagged as anomaly:             {stats['normal_flagged_as_anomaly']*100:.2f}%")

    print(f"\nRandom Noise ({args.noise_type}):")
    print(f"  Mean reconstruction probability: {stats['anomaly_mean_prob']:.6f}")
    print(f"  Std reconstruction probability:  {stats['anomaly_std_prob']:.6f}")
    print(f"  Flagged as anomaly:             {stats['anomaly_flagged_as_anomaly']*100:.2f}%")

    print("\n" + "="*70)
    print(f"Detection Rate: {stats['anomaly_flagged_as_anomaly']*100:.2f}% of noise detected")
    print(f"False Positive Rate: {stats['normal_flagged_as_anomaly']*100:.2f}% of normal flagged")
    print("="*70 + "\n")

    # Visualize results
    print("Creating visualizations...")
    visualize_results(model, normal_data, noise_data, num_samples=5)
    plot_probability_distributions(stats)

    print("\nDone!")


if __name__ == '__main__':
    main()
