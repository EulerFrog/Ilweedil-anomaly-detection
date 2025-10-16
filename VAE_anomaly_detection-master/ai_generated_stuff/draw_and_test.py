import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
import os

from model.VAE import VAEAnomalyTabular


class DrawingApp:
    def __init__(self, model, device, alpha=1.0):
        self.model = model
        self.device = device
        self.alpha = alpha

        # Create figure with drawing canvas and result display
        self.fig = plt.figure(figsize=(14, 6))
        self.fig.suptitle('Draw a Digit and Test Anomaly Detection', fontsize=16)

        # Drawing canvas (left)
        self.ax_draw = self.fig.add_subplot(2, 4, (1, 6))
        self.ax_draw.set_title('Draw Here (Click and Drag)', fontsize=12)
        self.ax_draw.set_xlim(0, 28)
        self.ax_draw.set_ylim(0, 28)
        self.ax_draw.set_aspect('equal')
        self.ax_draw.invert_yaxis()

        # Initialize canvas (starts black)
        self.canvas = np.zeros((28, 28))
        self.im = self.ax_draw.imshow(self.canvas, cmap='gray', vmin=0, vmax=1)  # Changed to 'gray' for black background

        # Results display
        self.ax_input = self.fig.add_subplot(2, 4, 3)
        self.ax_input.set_title('Preprocessed Input', fontsize=10)
        self.ax_input.axis('off')

        self.ax_recon = self.fig.add_subplot(2, 4, 4)
        self.ax_recon.set_title('VAE Reconstruction', fontsize=10)
        self.ax_recon.axis('off')

        self.ax_result = self.fig.add_subplot(2, 4, (7, 8))
        self.ax_result.axis('off')
        self.result_text = self.ax_result.text(0.5, 0.5, 'Draw a digit and click "Test"',
                                               ha='center', va='center', fontsize=14,
                                               bbox=dict(boxstyle='round', facecolor='lightgray'))

        # Buttons
        ax_test = self.fig.add_axes([0.55, 0.05, 0.12, 0.05])
        ax_clear = self.fig.add_axes([0.68, 0.05, 0.12, 0.05])
        ax_quit = self.fig.add_axes([0.81, 0.05, 0.12, 0.05])

        self.btn_test = Button(ax_test, 'Test')
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_quit = Button(ax_quit, 'Quit')

        self.btn_test.on_clicked(self.test_digit)
        self.btn_clear.on_clicked(self.clear_canvas)
        self.btn_quit.on_clicked(self.quit_app)

        # Mouse events for drawing
        self.drawing = False
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes == self.ax_draw:
            self.drawing = True
            self.draw_point(event.xdata, event.ydata)

    def on_release(self, event):
        self.drawing = False

    def on_motion(self, event):
        if self.drawing and event.inaxes == self.ax_draw:
            self.draw_point(event.xdata, event.ydata)

    def draw_point(self, x, y):
        if x is None or y is None:
            return

        # Draw with a brush (3x3 area)
        cx, cy = int(x), int(y)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                px, py = cx + dx, cy + dy
                if 0 <= px < 28 and 0 <= py < 28:
                    # Add to canvas with some transparency for smoother drawing
                    self.canvas[py, px] = min(1.0, self.canvas[py, px] + 0.3)

        self.im.set_data(self.canvas)
        self.fig.canvas.draw_idle()

    def clear_canvas(self, event):
        self.canvas = np.zeros((28, 28))
        self.im.set_data(self.canvas)

        # Clear result displays
        self.ax_input.clear()
        self.ax_input.set_title('Preprocessed Input', fontsize=10)
        self.ax_input.axis('off')

        self.ax_recon.clear()
        self.ax_recon.set_title('VAE Reconstruction', fontsize=10)
        self.ax_recon.axis('off')

        self.result_text.set_text('Draw a digit and click "Test"')
        self.result_text.set_bbox(dict(boxstyle='round', facecolor='lightgray'))

        self.fig.canvas.draw_idle()

    def preprocess_drawing(self):
        # Already white on black background (like MNIST), no inversion needed
        img = self.canvas

        # Flatten to 784-dim vector
        img_flat = img.flatten()

        return img_flat

    def test_digit(self, event):
        # Check if canvas is empty
        if self.canvas.sum() < 0.1:
            self.result_text.set_text('Please draw something first!')
            self.result_text.set_bbox(dict(boxstyle='round', facecolor='orange'))
            self.fig.canvas.draw_idle()
            return

        # Preprocess drawing
        img_flat = self.preprocess_drawing()

        # Convert to tensor
        input_tensor = torch.FloatTensor(img_flat).unsqueeze(0).to(self.device)

        # Get reconstruction probability
        self.model.eval()
        with torch.no_grad():
            recon_prob = self.model.reconstructed_probability(input_tensor).item()
            is_anomaly = self.model.is_anomaly(input_tensor, alpha=self.alpha).item()

            # Get reconstruction
            pred = self.model.predict(input_tensor)
            recon = pred['recon_mu'].mean(dim=0).cpu().numpy().reshape(28, 28)

        # Display preprocessed input
        self.ax_input.clear()
        self.ax_input.imshow(img_flat.reshape(28, 28), cmap='gray')
        self.ax_input.set_title('Preprocessed Input', fontsize=10)
        self.ax_input.axis('off')

        # Display reconstruction
        self.ax_recon.clear()
        self.ax_recon.imshow(recon, cmap='gray')
        self.ax_recon.set_title('VAE Reconstruction', fontsize=10)
        self.ax_recon.axis('off')

        # Display result
        if is_anomaly:
            result_str = f'ANOMALY DETECTED!\n\nReconstruction Probability: {recon_prob:.4f}\nThreshold: {self.alpha:.4f}\n\nThis does not look like a normal MNIST digit.'
            bgcolor = 'lightcoral'
        else:
            result_str = f'NORMAL DIGIT\n\nReconstruction Probability: {recon_prob:.4f}\nThreshold: {self.alpha:.4f}\n\nThis looks like a valid MNIST digit.'
            bgcolor = 'lightgreen'

        self.result_text.set_text(result_str)
        self.result_text.set_bbox(dict(boxstyle='round', facecolor=bgcolor))

        self.fig.canvas.draw_idle()

        print("\n" + "="*60)
        print("RESULT:")
        print(f"  Reconstruction Probability: {recon_prob:.6f}")
        print(f"  Threshold (alpha):          {self.alpha:.6f}")
        print(f"  Classification:             {'ANOMALY' if is_anomaly else 'NORMAL'}")
        print("="*60 + "\n")

    def quit_app(self, event):
        plt.close(self.fig)

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Interactive digit drawing and anomaly detection')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='Threshold for anomaly detection (default: 1.0)'
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
    latent_size = 64  # Updated to match the 100-epoch trained model
    num_resamples = 10

    # Create model instance
    model = VAEAnomalyTabular(input_size, latent_size, L=num_resamples)
    print(model)

    path = '../saved_models/' + args.checkpoint

    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  Input size: {model.input_size}")
    print(f"  Latent size: {model.latent_size}")
    print(f"  Anomaly threshold (alpha): {args.alpha}")

    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("  1. Click and drag on the left canvas to draw a digit")
    print("  2. Click 'Test' to check if it's anomalous")
    print("  3. Click 'Clear' to start over")
    print("  4. Click 'Quit' when done")
    print("="*60 + "\n")

    # Create and show drawing app
    app = DrawingApp(model, device, alpha=args.alpha)
    app.show()


if __name__ == '__main__':
    main()
