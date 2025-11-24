#!/usr/bin/env python3
"""
Download medical prescription dataset from HuggingFace for validation.

Dataset: chinmays18/medical-prescription-dataset
Downloads test split images and annotations to data/validation_dataset/
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

# Dataset configuration
REPO_ID = "chinmays18/medical-prescription-dataset"
DATASET_DIR = Path(__file__).parent.parent.parent / "data" / "validation_dataset"
IMAGES_DIR = DATASET_DIR / "test" / "images"
ANNOTATIONS_DIR = DATASET_DIR / "test" / "annotations"


def ensure_directories():
    """Create necessary directories if they don't exist."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directories under {DATASET_DIR}")


def download_files(repo_id: str, file_paths: list, local_dir: Path, subfolder: str = ""):
    """
    Download multiple files from HuggingFace repository.
    
    Args:
        repo_id: HuggingFace repository ID
        file_paths: List of file paths to download
        local_dir: Local directory to save files
        subfolder: Subfolder in the repository
    """
    print(f"\nDownloading {len(file_paths)} files to {local_dir}...")
    
    for file_path in tqdm(file_paths, desc="Downloading"):
        try:
            # Extract just the filename from the path
            filename = Path(file_path).name
            local_path = local_dir / filename
            
            # Skip if file already exists
            if local_path.exists():
                continue
            
            # Download file
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type="dataset",
                local_dir=local_dir.parent.parent,
                local_dir_use_symlinks=False
            )
            
            # Move to correct location if needed
            if Path(downloaded_path) != local_path:
                Path(downloaded_path).rename(local_path)
                
        except Exception as e:
            print(f"\n⚠ Error downloading {file_path}: {e}")
            continue


def get_test_files(repo_id: str):
    """
    Get lists of test images and annotations from the repository.
    
    Args:
        repo_id: HuggingFace repository ID
        
    Returns:
        Tuple of (image_files, annotation_files)
    """
    print("Fetching repository file list...")
    
    try:
        all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        
        # Filter test split files
        test_images = [f for f in all_files if f.startswith("test/images/") and f.endswith((".jpg", ".png", ".jpeg"))]
        test_annotations = [f for f in all_files if f.startswith("test/annotations/") and f.endswith(".txt")]
        
        print(f"✓ Found {len(test_images)} test images")
        print(f"✓ Found {len(test_annotations)} test annotations")
        
        return test_images, test_annotations
        
    except Exception as e:
        print(f"✗ Error fetching file list: {e}")
        sys.exit(1)


def verify_download():
    """Verify that files were downloaded successfully."""
    image_count = len(list(IMAGES_DIR.glob("*.[jp][pn][g]*")))
    annotation_count = len(list(ANNOTATIONS_DIR.glob("*.txt")))
    
    print(f"\n{'='*60}")
    print(f"Download Summary:")
    print(f"{'='*60}")
    print(f"Images downloaded: {image_count}")
    print(f"Annotations downloaded: {annotation_count}")
    print(f"Location: {DATASET_DIR}")
    print(f"{'='*60}\n")
    
    if image_count == 0 or annotation_count == 0:
        print("⚠ Warning: Some files may not have been downloaded.")
        return False
    
    return True


def main():
    """Main download function."""
    print(f"Downloading validation dataset from HuggingFace")
    print(f"Repository: {REPO_ID}")
    print(f"Destination: {DATASET_DIR}\n")
    
    # Check if huggingface_hub is installed
    try:
        import huggingface_hub
    except ImportError:
        print("✗ Error: huggingface_hub is not installed.")
        print("Install it with: uv pip install huggingface_hub")
        sys.exit(1)
    
    # Create directories
    ensure_directories()
    
    # Get file lists
    test_images, test_annotations = get_test_files(REPO_ID)
    
    if not test_images or not test_annotations:
        print("✗ No test files found in repository.")
        sys.exit(1)
    
    # Download images
    download_files(REPO_ID, test_images, IMAGES_DIR)
    
    # Download annotations
    download_files(REPO_ID, test_annotations, ANNOTATIONS_DIR)
    
    # Verify download
    if verify_download():
        print("✓ Dataset downloaded successfully!")
        print(f"\nYou can now run validation with:")
        print(f"  python validate_dataset.py")
        print(f"\nOr specify custom paths:")
        print(f"  python validate_dataset.py --images-dir {IMAGES_DIR} --annotations-dir {ANNOTATIONS_DIR}")
    else:
        print("✗ Download completed with warnings. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
