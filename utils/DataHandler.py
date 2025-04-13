import os
import zipfile
import random
import logging
import gdown
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
from typing import Tuple


def download_dataset(data_dir: str, zip_url: str, zip_filename: str, root_dir: str) -> None:
    """
    Downloads and extracts the dataset if not already downloaded
    
    Args:
        data_dir (str): Directory where the dataset should be available
        zip_url (str): URL from which to download the dataset
        zip_filename (str): Filename for the downloaded zip
        root_dir (str): Directory where files are stored/extracted
    """
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        print(f"Created directory: {root_dir}")
        
    if not os.path.exists(data_dir):
        print(f"Downloading dataset from {zip_url} to {zip_filename}")
        try:
            gdown.download(zip_url, zip_filename, quiet=False)
            print(f"Extracting dataset...")
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(root_dir)
            print(f"Extraction complete. Dataset available at {data_dir}", )
        except Exception as e:
            print(f"Error downloading or extracting dataset:", str(e))
            raise
    else:
        print(f"Dataset already exists at {data_dir}")

def _sample_subset(dataset, fraction: float, random_seed: int = 42) -> Subset:
    """
    Returns a subset of the dataset with the specified fraction.
    
    Args:
        dataset: The full dataset.
        fraction (float): Fraction to sample.
        random_seed (int): Seed for reproducibility.
        
    Returns:
        Subset: A sampled subset.
    """
    random.seed(random_seed)
    dataset_len = len(dataset)
    subset_size = max(1, int(fraction * dataset_len))
    indices = random.sample(range(dataset_len), subset_size)
    return Subset(dataset, indices)

def create_full_data_loaders(dataset_root: str, transform: transforms.Compose, batch_size: int = 32,
                             random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for training, validation, and test splits using the full dataset.

    Args:
        dataset_root (str): Root directory of the dataset.
        transform (transforms.Compose): Transformations to apply.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 32.
        random_seed (int, optional): Seed for reproducibility. Defaults to 42.
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: DataLoaders for train, validation, and test sets.
    """
    split_dirs = ['train', 'eval', 'test']
    if all(os.path.exists(os.path.join(dataset_root, sub)) for sub in split_dirs):
        train_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'train'),
                                              transform=transform)
        val_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'eval'),
                                            transform=transform)
        test_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'test'),
                                             transform=transform)
        print(f"Using pre-split datasets: train {len(train_dataset)}, val {len(val_dataset)}, test {len(test_dataset)}")
    else:
        full_dataset = datasets.ImageFolder(root=dataset_root, transform=transform)
        total_len = len(full_dataset)
        train_len = int(0.7 * total_len)
        val_len = int(0.15 * total_len)
        test_len = total_len - train_len - val_len
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(random_seed)
        )
        print(f"Randomly split dataset:train{len(train_dataset)},val {len(val_dataset)}, test {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def create_tuning_data_loaders(dataset_root: str, transform: transforms.Compose, batch_size: int = 32,
                               subset_fraction: float = 0.1, random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for training, validation, and test splits,
    sampling a subset of the data for rapid hyperparameter tuning.

    Args:
        dataset_root (str): Root directory of the dataset.
        transform (transforms.Compose): Transformations to apply.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 32.
        subset_fraction (float, optional): Fraction of data to sample for tuning. Defaults to 0.1.
        random_seed (int, optional): Seed for reproducibility. Defaults to 42.
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: DataLoaders for tuning on train, val, and test subsets.
    """
    # load the full dataset split
    train_loader, val_loader, test_loader = create_full_data_loaders(dataset_root, transform, batch_size, random_seed)
    
    # create subsets from each split
    train_dataset = _sample_subset(train_loader.dataset, subset_fraction, random_seed)
    val_dataset = _sample_subset(val_loader.dataset, subset_fraction, random_seed)
    test_dataset = _sample_subset(test_loader.dataset, subset_fraction, random_seed)
    print(f"Created tuning data loaders with subset fraction: {subset_fraction}", )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Created subset datasets for hyperparameter tuning: train {len(train_dataset)}, val {len(val_dataset)}, test {len(test_dataset)}")

    return train_loader, val_loader, test_loader