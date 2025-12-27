"""
Neural Navigator - Data Loader
Implements PyTorch Dataset for loading map images, text commands, and navigation paths.
"""

import os
import json
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class NavigationDataset(Dataset):
    """
    PyTorch Dataset for multimodal navigation path prediction.
    
    Loads:
    - 128x128 RGB map images
    - Text commands (tokenized)
    - Path coordinates (10 x,y pairs normalized to [0,1])
    
    Args:
        data_dir: Root directory containing images/ and annotations/ folders
        split: 'train' or 'test'
        transform: Optional image transformations
    """
    
    def __init__(
        self, 
        data_dir: str,
        split: str = 'train',
        transform=None
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        self.images_dir = os.path.join(data_dir, 'images')
        self.annotations_dir = os.path.join(data_dir, 'annotations')
        
        # Load metadata
        metadata_path = os.path.join(data_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.image_size = self.metadata['dataset_info']['image_size']
        self.num_path_points = self.metadata['dataset_info']['num_path_points']
        
        # Build vocabulary from text commands
        self.vocab = self._build_vocabulary()
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        
        # Store sample metadata (don't load all annotations at once)
        self.samples = self.metadata['samples']
        
        print(f"Loaded {len(self.samples)} samples from {split} set")
        print(f"Vocabulary size: {len(self.vocab)} tokens")
        
    def _build_vocabulary(self) -> List[str]:
        """Build vocabulary from all text commands in the dataset."""
        vocab_set = set()
        
        # Scan all annotation files to extract unique tokens
        for sample in self.metadata['samples']:
            # Parse text command
            text = sample['text']
            tokens = text.lower().split()
            vocab_set.update(tokens)
        
        # Add special tokens
        vocab = ['<PAD>', '<UNK>'] + sorted(list(vocab_set))
        return vocab
    
    def _load_annotation(self, idx: int) -> Dict:
        """Load a single annotation file lazily."""
        sample = self.samples[idx]
        annotation_file = sample['annotation_file']
        annotation_path = os.path.join(self.annotations_dir, annotation_file)
        
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        
        return annotation
    
    def _tokenize(self, text: str) -> List[int]:
        """Convert text command to list of token indices."""
        tokens = text.lower().split()
        token_ids = [
            self.word_to_idx.get(token, self.word_to_idx['<UNK>']) 
            for token in tokens
        ]
        return token_ids
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a sample with:
        - image: [3, 128, 128] normalized to [0,1]
        - tokens: [seq_len] token indices
        - path: [20] coordinates normalized to [0,1] (or None for test set)
        - text: original text string
        - image_file: filename
        """
        annotation = self.annotations[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, annotation['image_file'])
        image = Image.open(image_path).convert('RGB')
        
        # Convert to tensor and normalize to [0, 1]
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        
        if self.transform:
            image = self.transform(image)
        
        # Tokenize text
        text = annotation['text']
        token_ids = self._tokenize(text)
        tokens = torch.LongTensor(token_ids)
        
        # Prepare output dictionary
        sample = {
            'image': image,
            'tokens': tokens,
            'text': text,
            'image_file': annotation['image_file'],
            'id': annotation['id']
        }
        
        # Load path coordinates (only for training data)
        if 'path' in annotation:
            path = annotation['path']
            # Flatten to [20] and normalize to [0, 1]
            path_array = np.array(path, dtype=np.float32).flatten()
            path_normalized = path_array / self.image_size  # Normalize by image size
            sample['path'] = torch.from_numpy(path_normalized)
        else:
            # Test set - no ground truth path
            sample['path'] = None
        
        return sample


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable-length text sequences.
    Pads all sequences to the max length in the batch.
    """
    # Stack images
    images = torch.stack([item['image'] for item in batch])
    
    # Pad tokens to max length in batch
    token_sequences = [item['tokens'] for item in batch]
    max_len = max(len(seq) for seq in token_sequences)
    
    padded_tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(token_sequences):
        padded_tokens[i, :len(seq)] = seq
    
    # Stack paths (if available)
    if batch[0]['path'] is not None:
        paths = torch.stack([item['path'] for item in batch])
    else:
        paths = None
    
    return {
        'image': images,
        'tokens': padded_tokens,
        'path': paths,
        'text': [item['text'] for item in batch],
        'image_file': [item['image_file'] for item in batch],
        'id': [item['id'] for item in batch]
    }


def get_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 0
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        train_dir: Path to training data directory
        test_dir: Path to test data directory
        batch_size: Batch size
        val_split: Fraction of training data to use for validation
        num_workers: Number of data loading workers
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Load full training dataset
    full_dataset = NavigationDataset(train_dir, split='train')
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    # Load test dataset
    test_dataset = NavigationDataset(test_dir, split='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader, full_dataset.vocab


if __name__ == "__main__":
    # Test the data loader
    print("Testing NavigationDataset...")
    
    train_dir = "../data"
    dataset = NavigationDataset(train_dir, split='train')
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Vocabulary: {dataset.vocab}")
    
    # Get first sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"  Text: {sample['text']}")
    print(f"  Tokens: {sample['tokens']}")
    print(f"  Path shape: {sample['path'].shape if sample['path'] is not None else None}")
    print(f"  Path range: [{sample['path'].min():.3f}, {sample['path'].max():.3f}]" if sample['path'] is not None else "")
    
    # Test dataloader with batching
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    
    print(f"\nBatch test:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Tokens: {batch['tokens'].shape}")
    print(f"  Paths: {batch['path'].shape if batch['path'] is not None else None}")
