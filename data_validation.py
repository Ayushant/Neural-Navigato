"""
Data Sanity Checks & Validation Pipeline
Validates dataset quality before training to catch issues early.
"""

import os
import json
import numpy as np
from PIL import Image
from collections import Counter
from tqdm import tqdm


class DataValidator:
    """Comprehensive data quality checks for Neural Navigator dataset."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        self.annotations_dir = os.path.join(data_dir, 'annotations')
        self.metadata_path = os.path.join(data_dir, 'metadata.json')
        
        self.issues = []
        self.warnings = []
        self.stats = {}
        
    def run_all_checks(self):
        """Run all validation checks and generate report."""
        print("\n" + "="*60)
        print("NEURAL NAVIGATOR - DATA SANITY CHECKS")
        print("="*60)
        
        # 1. Metadata validation
        print("\n[1/8] Validating metadata.json...")
        self.check_metadata()
        
        # 2. File existence checks
        print("\n[2/8] Checking file existence...")
        self.check_files_exist()
        
        # 3. Image quality checks
        print("\n[3/8] Validating image properties...")
        self.check_image_quality()
        
        # 4. Annotation format checks
        print("\n[4/8] Validating annotation format...")
        self.check_annotation_format()
        
        # 5. Path coordinate checks
        print("\n[5/8] Validating path coordinates...")
        self.check_path_coordinates()
        
        # 6. Text command checks
        print("\n[6/8] Validating text commands...")
        self.check_text_commands()
        
        # 7. Data distribution analysis
        print("\n[7/8] Analyzing data distribution...")
        self.analyze_distribution()
        
        # 8. Cross-validation checks
        print("\n[8/8] Running cross-validation checks...")
        self.cross_validation_checks()
        
        # Generate report
        self.generate_report()
        
        return len(self.issues) == 0
    
    def check_metadata(self):
        """Validate metadata.json structure."""
        if not os.path.exists(self.metadata_path):
            self.issues.append("metadata.json not found!")
            return
        
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check required fields
        required_keys = ['dataset_info', 'samples']
        for key in required_keys:
            if key not in metadata:
                self.issues.append(f"Missing required key in metadata: {key}")
        
        # Validate dataset_info
        if 'dataset_info' in metadata:
            info = metadata['dataset_info']
            expected_size = info.get('image_size', 128)
            num_samples = info.get('num_samples', 0)
            num_path_points = info.get('num_path_points', 10)
            
            self.stats['expected_image_size'] = expected_size
            self.stats['declared_samples'] = num_samples
            self.stats['expected_path_points'] = num_path_points
            
            print(f"  ✓ Expected: {num_samples} samples, {expected_size}x{expected_size} images, {num_path_points} path points")
        
        self.metadata = metadata
    
    def check_files_exist(self):
        """Check if all declared files actually exist."""
        missing_images = []
        missing_annotations = []
        
        for sample in tqdm(self.metadata['samples'], desc="  Checking files"):
            image_path = os.path.join(self.images_dir, sample['image_file'])
            anno_path = os.path.join(self.annotations_dir, sample['annotation_file'])
            
            if not os.path.exists(image_path):
                missing_images.append(sample['image_file'])
            
            if not os.path.exists(anno_path):
                missing_annotations.append(sample['annotation_file'])
        
        if missing_images:
            self.issues.append(f"Missing {len(missing_images)} images: {missing_images[:5]}...")
        else:
            print(f"  ✓ All {len(self.metadata['samples'])} image files found")
        
        if missing_annotations:
            self.issues.append(f"Missing {len(missing_annotations)} annotations: {missing_annotations[:5]}...")
        else:
            print(f"  ✓ All {len(self.metadata['samples'])} annotation files found")
    
    def check_image_quality(self):
        """Validate image properties (size, channels, format)."""
        expected_size = self.stats['expected_image_size']
        
        wrong_size = []
        wrong_mode = []
        corrupted = []
        
        # Sample 100 images for quick check
        sample_indices = np.random.choice(len(self.metadata['samples']), 
                                         min(100, len(self.metadata['samples'])), 
                                         replace=False)
        
        for idx in tqdm(sample_indices, desc="  Validating images"):
            sample = self.metadata['samples'][idx]
            image_path = os.path.join(self.images_dir, sample['image_file'])
            
            try:
                img = Image.open(image_path)
                
                # Check dimensions
                if img.size != (expected_size, expected_size):
                    wrong_size.append((sample['image_file'], img.size))
                
                # Check color mode
                if img.mode != 'RGB':
                    wrong_mode.append((sample['image_file'], img.mode))
                
            except Exception as e:
                corrupted.append((sample['image_file'], str(e)))
        
        if wrong_size:
            self.issues.append(f"{len(wrong_size)} images have wrong size: {wrong_size[:3]}")
        else:
            print(f"  ✓ All sampled images are {expected_size}x{expected_size}")
        
        if wrong_mode:
            self.warnings.append(f"{len(wrong_mode)} images not in RGB mode: {wrong_mode[:3]}")
        else:
            print(f"  ✓ All sampled images are RGB")
        
        if corrupted:
            self.issues.append(f"{len(corrupted)} corrupted images: {corrupted}")
        else:
            print(f"  ✓ No corrupted images found")
    
    def check_annotation_format(self):
        """Validate annotation JSON structure."""
        required_fields = ['id', 'image_file', 'text', 'target']
        
        invalid_annotations = []
        
        # Check all annotations
        for sample in tqdm(self.metadata['samples'], desc="  Checking annotations"):
            anno_path = os.path.join(self.annotations_dir, sample['annotation_file'])
            
            try:
                with open(anno_path, 'r') as f:
                    anno = json.load(f)
                
                # Check required fields (path is optional for test data)
                missing = [field for field in required_fields if field not in anno]
                if missing:
                    invalid_annotations.append((sample['annotation_file'], f"Missing fields: {missing}"))
                
            except json.JSONDecodeError as e:
                invalid_annotations.append((sample['annotation_file'], f"Invalid JSON: {e}"))
            except Exception as e:
                invalid_annotations.append((sample['annotation_file'], str(e)))
        
        if invalid_annotations:
            self.issues.append(f"{len(invalid_annotations)} invalid annotations: {invalid_annotations[:3]}")
        else:
            print(f"  ✓ All annotations have correct format")
    
    def check_path_coordinates(self):
        """Validate path coordinates are within bounds."""
        expected_points = self.stats['expected_path_points']
        image_size = self.stats['expected_image_size']
        
        wrong_length = []
        out_of_bounds = []
        invalid_coords = []
        paths_checked = 0
        
        for sample in tqdm(self.metadata['samples'], desc="  Checking paths"):
            anno_path = os.path.join(self.annotations_dir, sample['annotation_file'])
            
            with open(anno_path, 'r') as f:
                anno = json.load(f)
            
            if 'path' not in anno:
                continue  # Test data doesn't have paths
            
            paths_checked += 1
            path = anno['path']
            
            # Check length
            if len(path) != expected_points:
                wrong_length.append((sample['annotation_file'], len(path)))
            
            # Check coordinates
            for i, (x, y) in enumerate(path):
                if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    invalid_coords.append((sample['annotation_file'], i, (x, y)))
                elif x < 0 or x > image_size or y < 0 or y > image_size:
                    out_of_bounds.append((sample['annotation_file'], i, (x, y)))
        
        if wrong_length:
            self.issues.append(f"{len(wrong_length)} paths have wrong length: {wrong_length[:3]}")
        else:
            print(f"  ✓ All {paths_checked} paths have {expected_points} points")
        
        if out_of_bounds:
            self.issues.append(f"{len(out_of_bounds)} coordinates out of bounds: {out_of_bounds[:3]}")
        else:
            print(f"  ✓ All coordinates within [0, {image_size}]")
        
        if invalid_coords:
            self.issues.append(f"{len(invalid_coords)} invalid coordinates: {invalid_coords[:3]}")
    
    def check_text_commands(self):
        """Validate text commands."""
        all_texts = []
        empty_texts = []
        
        for sample in self.metadata['samples']:
            text = sample.get('text', '')
            all_texts.append(text)
            
            if not text or not text.strip():
                empty_texts.append(sample['annotation_file'])
        
        if empty_texts:
            self.issues.append(f"{len(empty_texts)} empty text commands")
        else:
            print(f"  ✓ No empty text commands")
        
        # Vocabulary analysis
        all_words = []
        for text in all_texts:
            all_words.extend(text.lower().split())
        
        vocab = set(all_words)
        word_counts = Counter(all_words)
        
        self.stats['vocabulary_size'] = len(vocab)
        self.stats['vocabulary'] = sorted(vocab)
        self.stats['word_frequency'] = dict(word_counts)
        
        print(f"  ✓ Vocabulary size: {len(vocab)} unique words")
        print(f"    Most common: {word_counts.most_common(5)}")
    
    def analyze_distribution(self):
        """Analyze target distribution and path patterns."""
        shapes = []
        colors = []
        
        for sample in tqdm(self.metadata['samples'], desc="  Analyzing distribution"):
            anno_path = os.path.join(self.annotations_dir, sample['annotation_file'])
            
            with open(anno_path, 'r') as f:
                anno = json.load(f)
            
            if 'target' in anno:
                target = anno['target']
                shapes.append(target.get('shape', 'Unknown'))
                colors.append(target.get('color', 'Unknown'))
        
        shape_dist = Counter(shapes)
        color_dist = Counter(colors)
        
        self.stats['shape_distribution'] = dict(shape_dist)
        self.stats['color_distribution'] = dict(color_dist)
        
        print(f"  ✓ Shape distribution: {dict(shape_dist)}")
        print(f"  ✓ Color distribution: {dict(color_dist)}")
        
        # Check balance
        total = len(shapes)
        if total > 0:
            for shape, count in shape_dist.items():
                ratio = count / total
                if ratio < 0.2 or ratio > 0.5:
                    self.warnings.append(f"Imbalanced shape distribution: {shape} = {ratio:.1%}")
    
    def cross_validation_checks(self):
        """Check consistency between metadata and annotations."""
        mismatches = []
        
        sample_size = min(100, len(self.metadata['samples']))
        for sample in tqdm(self.metadata['samples'][:sample_size], desc="  Cross-validating"):
            anno_path = os.path.join(self.annotations_dir, sample['annotation_file'])
            
            with open(anno_path, 'r') as f:
                anno = json.load(f)
            
            # Check ID match
            if anno.get('id') != sample.get('id'):
                mismatches.append(f"ID mismatch in {sample['annotation_file']}")
            
            # Check image file match
            if anno.get('image_file') != sample.get('image_file'):
                mismatches.append(f"Image file mismatch in {sample['annotation_file']}")
            
            # Check text match
            if anno.get('text') != sample.get('text'):
                mismatches.append(f"Text mismatch in {sample['annotation_file']}")
        
        if mismatches:
            self.issues.append(f"{len(mismatches)} metadata-annotation mismatches")
        else:
            print(f"  ✓ Metadata and annotations are consistent")
    
    def generate_report(self):
        """Generate final validation report."""
        print("\n" + "="*60)
        print("VALIDATION REPORT")
        print("="*60)
        
        if not self.issues:
            print("\n✅ ALL CHECKS PASSED!")
            print("\nDataset is clean and ready for training.")
        else:
            print(f"\n❌ FOUND {len(self.issues)} CRITICAL ISSUES:")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
        
        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"Total samples: {self.stats.get('declared_samples', 'N/A')}")
        print(f"Image size: {self.stats.get('expected_image_size', 'N/A')}x{self.stats.get('expected_image_size', 'N/A')}")
        print(f"Path points: {self.stats.get('expected_path_points', 'N/A')}")
        print(f"Vocabulary: {self.stats.get('vocabulary_size', 'N/A')} unique words")
        print(f"  Words: {self.stats.get('vocabulary', [])}")
        print(f"Shape distribution: {self.stats.get('shape_distribution', {})}")
        print(f"Color distribution: {self.stats.get('color_distribution', {})}")
        print("="*60 + "\n")
        
        # Save report
        report_path = 'data_validation_report.json'
        with open(report_path, 'w') as f:
            json.dump({
                'issues': self.issues,
                'warnings': self.warnings,
                'stats': self.stats
            }, f, indent=2)
        
        print(f"📄 Detailed report saved to: {report_path}")


def main():
    """Run data validation on train and test datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Sanity Checks')
    parser.add_argument('--train_dir', type=str, default='../data',
                       help='Training data directory')
    parser.add_argument('--test_dir', type=str, default='../test_data',
                       help='Test data directory')
    args = parser.parse_args()
    
    # Validate training data
    print("\n🔍 VALIDATING TRAINING DATA")
    train_validator = DataValidator(args.train_dir)
    train_valid = train_validator.run_all_checks()
    
    # Validate test data
    print("\n🔍 VALIDATING TEST DATA")
    test_validator = DataValidator(args.test_dir)
    test_valid = test_validator.run_all_checks()
    
    if train_valid and test_valid:
        print("\n✅ Both datasets passed all validation checks!")
        return 0
    else:
        print("\n❌ Validation failed. Please fix issues before training.")
        return 1


if __name__ == "__main__":
    exit(main())
