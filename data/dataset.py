"""
PyTorch Dataset and DataLoader for EMOTHAW handwriting data.

Provides an image-based dataset that renders pen trajectories
as grayscale images for CNN classification.
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EmothawImageDataset(Dataset):
    """
    PyTorch Dataset for EMOTHAW handwriting trajectory images.

    Each sample is a grayscale image (1, H, W) rendered from pen trajectory data,
    paired with an emotion label.
    """

    def __init__(self, samples: list, labels: dict, target_emotion: str = "anxiety",
                 image_size: int = 224, cache_dir: str = None, transform=None):
        """
        Args:
            samples: List of sample dicts from svc_parser.load_all_svc_files().
            labels: Dict mapping user_id → int label.
            target_emotion: Which emotion dimension to use.
            image_size: Size for rendered images.
            cache_dir: Directory to cache rendered images.
            transform: Optional torchvision transforms.
        """
        self.image_size = image_size
        self.cache_dir = cache_dir
        self.transform = transform

        # Filter samples to only those with labels
        self.items = []
        for s in samples:
            uid = s["user_id"]
            if uid in labels:
                self.items.append({
                    "user_id": uid,
                    "task_id": s["task_id"],
                    "data": s["data"],
                    "label": labels[uid],
                    "filepath": s["filepath"],
                })

        print(f"Dataset: {len(self.items)} samples with labels "
              f"(from {len(set(it['user_id'] for it in self.items))} users)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # Try to load from cache
        img = None
        cache_path = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = os.path.join(
                self.cache_dir,
                f"u{item['user_id']:05d}_t{item['task_id']:05d}.png"
            )
            if os.path.exists(cache_path):
                import cv2
                img = cv2.imread(cache_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            from features.image_generator import trajectory_to_image
            img = trajectory_to_image(item["data"], image_size=self.image_size)
            if cache_path:
                import cv2
                cv2.imwrite(cache_path, img)

        # Convert to float tensor [0, 1] and add channel dimension
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # (1, H, W)

        # Stack to 3 channels for compatibility with pretrained models
        img_tensor = img_tensor.repeat(3, 1, 1)  # (3, H, W)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        label = torch.tensor(item["label"], dtype=torch.long)
        return img_tensor, label


def get_dataloaders(samples: list, labels: dict,
                    batch_size: int = 16,
                    image_size: int = 224,
                    cache_dir: str = None,
                    train_ratio: float = 0.7,
                    val_ratio: float = 0.15,
                    random_seed: int = 42):
    """
    Create train/val/test DataLoaders with user-level splitting to avoid data leakage.

    All samples from a given user go into the same split (train, val, or test),
    so the model never sees handwriting from a test user during training.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    from sklearn.model_selection import GroupShuffleSplit
    
    # Filter to samples with labels
    valid_samples = [s for s in samples if s["user_id"] in labels]
    user_ids = np.array([s["user_id"] for s in valid_samples])
    unique_users = np.unique(user_ids)

    # First split: train+val vs test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=1.0 - train_ratio, random_state=random_seed)
    train_val_idx, test_idx = next(gss1.split(valid_samples, groups=user_ids))

    # Second split: train vs val (from train+val set)
    train_val_samples = [valid_samples[i] for i in train_val_idx]
    train_val_user_ids = np.array([s["user_id"] for s in train_val_samples])

    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val_ratio, random_state=random_seed)
    train_idx_rel, val_idx_rel = next(gss2.split(train_val_samples, groups=train_val_user_ids))

    train_samples = [train_val_samples[i] for i in train_idx_rel]
    val_samples = [train_val_samples[i] for i in val_idx_rel]
    test_samples = [valid_samples[i] for i in test_idx]

    print(f"\nData split (by user, no leakage):")
    print(f"  Train: {len(train_samples)} samples ({len(set(s['user_id'] for s in train_samples))} users)")
    print(f"  Val:   {len(val_samples)} samples ({len(set(s['user_id'] for s in val_samples))} users)")
    print(f"  Test:  {len(test_samples)} samples ({len(set(s['user_id'] for s in test_samples))} users)")

    # Print class distribution per split
    for name, subset in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        label_counts = {}
        for s in subset:
            lbl = labels[s["user_id"]]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        print(f"  {name} class dist: {label_counts}")

    train_dataset = EmothawImageDataset(train_samples, labels, image_size=image_size, cache_dir=cache_dir)
    val_dataset = EmothawImageDataset(val_samples, labels, image_size=image_size, cache_dir=cache_dir)
    test_dataset = EmothawImageDataset(test_samples, labels, image_size=image_size, cache_dir=cache_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
