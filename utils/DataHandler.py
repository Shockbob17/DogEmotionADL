import os
import zipfile
import gdown
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple
from collections import Counter


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

def _stratified_subset(dataset, fraction: float, seed: int = 42) -> Subset:
    """
    Create a stratified subset of the dataset by preserving class distribution.

    Args:
        dataset: A torchvision dataset (e.g., ImageFolder).
        fraction (float): Fraction to sample from each class.
        seed (int): Random seed for reproducibility.

    Returns:
        torch.utils.data.Subset: A stratified subset of the dataset.
    """
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "samples"):
        labels = [s[1] for s in dataset.samples]
    else:
        raise AttributeError("Dataset must have 'targets' or 'samples' attribute.")

    subset_split = StratifiedShuffleSplit(n_splits=1, test_size=1 - fraction, random_state=seed)
    indices, _ = next(subset_split.split(range(len(labels)), labels))
    return Subset(dataset, indices)

def _show_class_distribution(dataset, dataset_desc:str = "Training") -> None:
    """
    Print number of samples per class in a Subset dataset

    Args:
        dataset: A torch.utils.data.Subset wrapping dataset
    """
    try:
        indices = dataset.indices
        base_dataset = dataset.dataset
        targets = [base_dataset.targets[i] for i in indices]
        classes = base_dataset.classes
        label_counts = Counter(targets)

        print(f"Class Distribution for {dataset_desc}:")
        for idx, class_name in enumerate(classes):
            print(f"  {class_name:<10}: {label_counts.get(idx, 0)}")

    except AttributeError as e:
        print("Failed to access targets/classes from base dataset.")
        print(f"Reason: {e}")
    except Exception as e:
        print("Unexpected error during class distribution analysis.")
        print(f"Reason: {e}")

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
    # apply stratified sampling to train and val
    train_subset = _stratified_subset(train_loader.dataset, subset_fraction, seed=random_seed)
    val_subset = _stratified_subset(val_loader.dataset, subset_fraction, seed=random_seed)
    
    print(f"Created tuning data loaders with subset fraction: {subset_fraction}", )
    # full test set is kept
    test_dataset = test_loader.dataset
    
    print(f"Created subset datasets for hyperparameter tuning: train {len(train_subset)}, val {len(val_subset)}, test {len(test_dataset)}")
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    _show_class_distribution(train_loader.dataset, "Subset Training")
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    _show_class_distribution(val_loader.dataset, "Subset Validation")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader