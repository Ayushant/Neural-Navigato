"""
Neural Navigator - Multimodal Model Architecture
Combines vision (CNN) and text (embeddings) to predict navigation paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoder(nn.Module):
    """
    Lightweight CNN for encoding 128x128 RGB map images.
    Outputs a fixed-size feature vector.
    """
    
    def __init__(self, output_dim: int = 256):
        super().__init__()
        
        # CNN layers: progressively downsample from 128x128
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)  # 64x64
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 32x32
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 16x16
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 8x8
        self.bn4 = nn.BatchNorm2d(256)
        
        # Global average pooling + projection
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, 3, 128, 128] image tensor
        Returns:
            [batch_size, output_dim] feature vector
        """
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global pooling and projection
        x = self.pool(x)  # [batch, 256, 1, 1]
        x = x.flatten(1)  # [batch, 256]
        x = self.fc(x)     # [batch, output_dim]
        
        return x


class TextEncoder(nn.Module):
    """
    Simple learnable embedding layer for text commands.
    Uses vocabulary of ~10 unique tokens.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 64, output_dim: int = 128):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, tokens):
        """
        Args:
            tokens: [batch_size, seq_len] token indices
        Returns:
            [batch_size, output_dim] text feature vector
        """
        # Embed tokens
        embedded = self.embedding(tokens)  # [batch, seq_len, embedding_dim]
        
        # Mean pooling over sequence
        pooled = embedded.mean(dim=1)  # [batch, embedding_dim]
        
        # Project to output dimension
        output = self.fc(pooled)  # [batch, output_dim]
        
        return output


class PathDecoder(nn.Module):
    """
    MLP decoder that outputs 20 values (10 x,y coordinate pairs).
    Takes fused multimodal features as input.
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [512, 256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer: 20 values (10 x,y pairs), normalized to [0, 1]
        layers.append(nn.Linear(prev_dim, 20))
        layers.append(nn.Sigmoid())  # Constrain output to [0, 1]
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim] fused features
        Returns:
            [batch_size, 20] predicted path coordinates (normalized)
        """
        return self.decoder(x)


class NeuralNavigator(nn.Module):
    """
    Complete multimodal model for navigation path prediction.
    
    Architecture:
    1. Vision encoder (CNN) processes map image
    2. Text encoder (embeddings) processes command
    3. Fusion layer concatenates features
    4. Path decoder (MLP) outputs 20 coordinates
    """
    
    def __init__(
        self, 
        vocab_size: int,
        vision_dim: int = 256,
        text_dim: int = 128,
        decoder_hidden: list = [512, 256, 128]
    ):
        super().__init__()
        
        self.vision_encoder = VisionEncoder(output_dim=vision_dim)
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=64,
            output_dim=text_dim
        )
        
        # Fusion dimension
        fusion_dim = vision_dim + text_dim
        
        self.path_decoder = PathDecoder(
            input_dim=fusion_dim,
            hidden_dims=decoder_hidden
        )
        
    def forward(self, image, tokens):
        """
        Args:
            image: [batch_size, 3, 128, 128] map images
            tokens: [batch_size, seq_len] text token indices
        Returns:
            [batch_size, 20] predicted path coordinates (normalized to [0,1])
        """
        # Encode image and text
        vision_features = self.vision_encoder(image)  # [batch, vision_dim]
        text_features = self.text_encoder(tokens)     # [batch, text_dim]
        
        # Fuse features by concatenation
        fused = torch.cat([vision_features, text_features], dim=1)  # [batch, vision_dim + text_dim]
        
        # Decode to path coordinates
        path = self.path_decoder(fused)  # [batch, 20]
        
        return path
    
    def predict_path(self, image, tokens):
        """
        Inference method that returns path as list of (x, y) tuples.
        Coordinates are denormalized to [0, 128] pixel space.
        
        Args:
            image: [batch_size, 3, 128, 128] or [3, 128, 128]
            tokens: [batch_size, seq_len] or [seq_len]
        Returns:
            List of [(x1, y1), (x2, y2), ..., (x10, y10)] for each sample
        """
        # Handle single sample case
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        
        # Get predictions
        with torch.no_grad():
            path_normalized = self.forward(image, tokens)  # [batch, 20]
        
        # Denormalize to [0, 128] pixel coordinates
        path_pixels = path_normalized * 128.0
        
        # Reshape to [(x, y), ...] format
        batch_size = path_pixels.shape[0]
        paths = []
        
        for i in range(batch_size):
            coords = path_pixels[i].reshape(10, 2).cpu().numpy()
            path_list = [(float(x), float(y)) for x, y in coords]
            paths.append(path_list)
        
        return paths if batch_size > 1 else paths[0]


def create_model(vocab_size: int, device: str = 'cuda') -> NeuralNavigator:
    """
    Factory function to create and initialize the model.
    
    Args:
        vocab_size: Size of text vocabulary
        device: Device to place model on ('cuda' or 'cpu')
    Returns:
        Initialized NeuralNavigator model
    """
    model = NeuralNavigator(
        vocab_size=vocab_size,
        vision_dim=256,
        text_dim=128,
        decoder_hidden=[512, 256, 128]
    )
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.01)
    
    model.apply(init_weights)
    model = model.to(device)
    
    return model


if __name__ == "__main__":
    # Test model architecture
    print("Testing NeuralNavigator model...")
    
    vocab_size = 12  # Example vocabulary size
    batch_size = 4
    
    # Create model
    model = create_model(vocab_size, device='cpu')
    
    # Create dummy inputs
    dummy_image = torch.randn(batch_size, 3, 128, 128)
    dummy_tokens = torch.randint(0, vocab_size, (batch_size, 6))
    
    # Forward pass
    output = model(dummy_image, dummy_tokens)
    
    print(f"\nModel Architecture:")
    print(f"  Input image: {dummy_image.shape}")
    print(f"  Input tokens: {dummy_tokens.shape}")
    print(f"  Output path: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Test prediction method
    paths = model.predict_path(dummy_image[0], dummy_tokens[0])
    print(f"\nPredicted path (single sample): {len(paths)} points")
    print(f"  First point: {paths[0]}")
    print(f"  Last point: {paths[-1]}")
