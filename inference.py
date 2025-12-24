"""
Neural Navigator - Inference Script
Loads trained model and generates path predictions with visualizations.
"""

import os
import json
import argparse
from typing import List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from tqdm import tqdm

from utils.data_loader import NavigationDataset, collate_fn
from models.model import NeuralNavigator


def load_model(checkpoint_path: str, device: str = 'cuda') -> Tuple[NeuralNavigator, List[str]]:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        (model, vocabulary)
    """
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = checkpoint['vocab']
    vocab_size = len(vocab)
    
    # Create model architecture
    model = NeuralNavigator(
        vocab_size=vocab_size,
        vision_dim=256,
        text_dim=128,
        decoder_hidden=[512, 256, 128]
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"  Vocabulary size: {vocab_size}")
    
    return model, vocab


def visualize_prediction(
    image: np.ndarray,
    predicted_path: List[Tuple[float, float]],
    ground_truth_path: List[Tuple[float, float]] = None,
    text_command: str = "",
    save_path: str = None
):
    """
    Visualize predicted path overlaid on the map image.
    
    Args:
        image: RGB image array [H, W, 3]
        predicted_path: List of (x, y) coordinate tuples
        ground_truth_path: Optional ground truth path for comparison
        text_command: Text command for title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Display image
    ax.imshow(image)
    
    # Extract coordinates
    pred_x = [p[0] for p in predicted_path]
    pred_y = [p[1] for p in predicted_path]
    
    # Plot predicted path
    ax.plot(pred_x, pred_y, 'b-', linewidth=2.5, label='Predicted Path', alpha=0.8)
    ax.plot(pred_x, pred_y, 'bo', markersize=8, alpha=0.6)
    
    # Mark start and end
    ax.plot(pred_x[0], pred_y[0], 'go', markersize=12, label='Start', zorder=10)
    ax.plot(pred_x[-1], pred_y[-1], 'ro', markersize=12, label='Target', zorder=10)
    
    # Add arrow to show direction
    if len(predicted_path) > 1:
        arrow = FancyArrowPatch(
            (pred_x[-2], pred_y[-2]), (pred_x[-1], pred_y[-1]),
            arrowstyle='->', mutation_scale=20, linewidth=2.5,
            color='red', alpha=0.7
        )
        ax.add_patch(arrow)
    
    # Plot ground truth if available
    if ground_truth_path is not None:
        gt_x = [p[0] for p in ground_truth_path]
        gt_y = [p[1] for p in ground_truth_path]
        ax.plot(gt_x, gt_y, 'r--', linewidth=2, label='Ground Truth', alpha=0.5)
        ax.plot(gt_x, gt_y, 'r^', markersize=6, alpha=0.4)
    
    # Formatting
    ax.set_xlim(0, 128)
    ax.set_ylim(128, 0)  # Invert y-axis for image coordinates
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title(f'Command: "{text_command}"', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def predict_and_visualize(
    model: NeuralNavigator,
    dataset: NavigationDataset,
    output_dir: str,
    device: str = 'cuda',
    max_samples: int = None,
    save_images: bool = True
):
    """
    Run inference on dataset and generate visualizations.
    
    Args:
        model: Trained model
        dataset: Dataset to run inference on
        output_dir: Directory to save outputs
        device: Device for inference
        max_samples: Maximum number of samples to process (None for all)
        save_images: Whether to save visualization images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    results = []
    
    print(f"\nGenerating predictions for {num_samples} samples...")
    
    with torch.no_grad():
        for idx in tqdm(range(num_samples)):
            sample = dataset[idx]
            
            # Prepare inputs
            image = sample['image'].unsqueeze(0).to(device)  # [1, 3, 128, 128]
            tokens = sample['tokens'].unsqueeze(0).to(device)  # [1, seq_len]
            
            # Get prediction
            predicted_path = model.predict_path(image, tokens)
            
            # Convert to list of tuples
            pred_coords = [(float(x), float(y)) for x, y in predicted_path]
            
            # Get ground truth if available
            gt_coords = None
            if sample['path'] is not None:
                gt_array = sample['path'].numpy() * 128.0  # Denormalize
                gt_coords = [(float(gt_array[i]), float(gt_array[i+1])) 
                            for i in range(0, len(gt_array), 2)]
            
            # Store results
            result = {
                'id': sample['id'],
                'image_file': sample['image_file'],
                'text': sample['text'],
                'predicted_path': pred_coords
            }
            
            if gt_coords is not None:
                result['ground_truth_path'] = gt_coords
                
                # Calculate error metrics
                pred_array = np.array(pred_coords)
                gt_array = np.array(gt_coords)
                mse = np.mean((pred_array - gt_array) ** 2)
                mae = np.mean(np.abs(pred_array - gt_array))
                
                result['mse'] = float(mse)
                result['mae'] = float(mae)
            
            results.append(result)
            
            # Visualize and save
            if save_images:
                # Load original image
                image_path = os.path.join(dataset.images_dir, sample['image_file'])
                orig_image = np.array(Image.open(image_path))
                
                # Create visualization
                save_path = os.path.join(
                    output_dir, 
                    f"prediction_{sample['id']:06d}.png"
                )
                
                visualize_prediction(
                    image=orig_image,
                    predicted_path=pred_coords,
                    ground_truth_path=gt_coords,
                    text_command=sample['text'],
                    save_path=save_path
                )
    
    # Save predictions to JSON
    predictions_path = os.path.join(output_dir, 'predictions.json')
    with open(predictions_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nPredictions saved to {predictions_path}")
    
    # Calculate and print statistics if ground truth available
    if results[0].get('mse') is not None:
        mse_values = [r['mse'] for r in results]
        mae_values = [r['mae'] for r in results]
        
        print(f"\nEvaluation Metrics:")
        print(f"  Average MSE: {np.mean(mse_values):.4f} (±{np.std(mse_values):.4f})")
        print(f"  Average MAE: {np.mean(mae_values):.4f} (±{np.std(mae_values):.4f})")
        
        # Save metrics
        metrics = {
            'num_samples': len(results),
            'mean_mse': float(np.mean(mse_values)),
            'std_mse': float(np.std(mse_values)),
            'mean_mae': float(np.mean(mae_values)),
            'std_mae': float(np.std(mae_values))
        }
        
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to {metrics_path}")
    
    return results


def main(args):
    """Main inference function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, vocab = load_model(args.checkpoint, device=device)
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = NavigationDataset(
        data_dir=args.data_dir,
        split=args.split
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create output directory
    output_dir = args.output_dir
    if args.split == 'test':
        output_dir = os.path.join(output_dir, 'test_predictions')
    else:
        output_dir = os.path.join(output_dir, 'validation_predictions')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run inference
    results = predict_and_visualize(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        device=device,
        max_samples=args.max_samples,
        save_images=args.save_images
    )
    
    print(f"\nInference completed! Results saved to {output_dir}")
    
    # Show a few sample predictions
    if args.show_samples > 0:
        print(f"\nShowing {args.show_samples} sample predictions...")
        for i in range(min(args.show_samples, len(results))):
            result = results[i]
            print(f"\nSample {result['id']}:")
            print(f"  Text: {result['text']}")
            print(f"  Predicted start: {result['predicted_path'][0]}")
            print(f"  Predicted end: {result['predicted_path'][-1]}")
            if 'ground_truth_path' in result:
                print(f"  GT start: {result['ground_truth_path'][0]}")
                print(f"  GT end: {result['ground_truth_path'][-1]}")
                print(f"  Error (MSE): {result['mse']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Navigator Inference')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='test_data',
                        help='Path to data directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                        help='Dataset split to run inference on')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='neural_navigator/outputs',
                        help='Directory to save predictions and visualizations')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (default: all)')
    parser.add_argument('--save_images', action='store_true', default=True,
                        help='Save visualization images')
    parser.add_argument('--show_samples', type=int, default=3,
                        help='Number of sample predictions to print')
    
    # System arguments
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU inference (default: use GPU if available)')
    
    args = parser.parse_args()
    
    main(args)
