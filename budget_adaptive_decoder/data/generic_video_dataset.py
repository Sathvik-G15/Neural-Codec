"""
Generic Video Dataset for Budget-Constrained Neural Decoder.

Works with ANY folder of video frames for development and testing.
Does NOT require Vimeo90k or any specific dataset format.

Expected structure (flexible):
    root/
        sequence_001/
            frame_001.png
            frame_002.png
            ...
        sequence_002/
            ...
    OR
    root/
        frame_001.png
        frame_002.png
        ... (single folder, all frames treated as one sequence)

Minimum: 2 consecutive frames anywhere.

This is for DEVELOPMENT ONLY. Replace with actual Vimeo90k for final training.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import glob
from typing import List, Tuple, Optional, Callable


class GenericVideoDataset(Dataset):
    """
    Generic video dataset that works with any folder structure.

    Returns (curr_frame, ref_frame) pairs where ref_frame is the temporal
    predecessor of curr_frame.

    Supports:
    - Multiple subfolders (sequence structure)
    - Single folder with all frames
    - Common image formats (.png, .jpg, .jpeg)
    """

    def __init__(
        self,
        root: str,
        height: int = 256,
        width: int = 448,
        corruption_sigma: float = 0.0,
        transform: Optional[Callable] = None,
        frame_pairs_per_sequence: int = 6,
    ):
        """
        Args:
            root: Root directory containing video frames
            height: Frame height (default 256 for Vimeo90k compatibility)
            width: Frame width (default 448 for Vimeo90k compatibility)
            corruption_sigma: Reference frame corruption strength (>0 for Phase 3/4)
            transform: Optional custom transform (overrides ToTensor + Resize)
            frame_pairs_per_sequence: How many P-frame pairs to extract per sequence
        """
        self.corruption_sigma = corruption_sigma
        self.frame_pairs_per_sequence = frame_pairs_per_sequence

        # Default transform: resize + convert to tensor
        if transform is None:
            self.transform = T.Compose([
                T.Resize((height, width)),
                T.ToTensor()
            ])
        else:
            self.transform = transform

        self.height = height
        self.width = width

        # Find all frame pairs
        self.pairs = self._find_pairs(root)

        if len(self.pairs) == 0:
            raise ValueError(
                f"No frame pairs found in {root}.\n"
                f"Need at least 2 sequential frames.\n"
                f"Supported formats: .png, .jpg, .jpeg"
            )

        print(f"GenericVideoDataset: {len(self.pairs)} frame pairs from {root}")

    def _find_pairs(self, root: str) -> List[Tuple[str, str]]:
        """Find all (curr, ref) frame pairs in the directory."""
        pairs = []
        root = os.path.expanduser(root)

        # Check for subfolders (sequence structure)
        subfolders = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and not d.startswith('.')
        ])

        if subfolders:
            # Multiple sequences
            for seq in subfolders:
                seq_path = os.path.join(root, seq)
                seq_pairs = self._find_pairs_in_folder(seq_path)
                pairs.extend(seq_pairs)
        else:
            # Single folder
            pairs = self._find_pairs_in_folder(root)

        return pairs

    def _find_pairs_in_folder(self, folder: str) -> List[Tuple[str, str]]:
        """Find frame pairs within a single folder."""
        # Find all image files. We deliberately only scan lowercase
        # extensions to avoid matching the same file twice on case-
        # insensitive filesystems (e.g., Windows: im1.png matches both
        # '*.png' and '*.PNG').
        extensions = ['*.png', '*.jpg', '*.jpeg']
        seen = set()
        frames = []
        for ext in extensions:
            for path in glob.glob(os.path.join(folder, ext)):
                norm = os.path.normcase(os.path.abspath(path))
                if norm in seen:
                    continue
                seen.add(norm)
                frames.append(path)

        if not frames:
            return []

        # Sort frames numerically if possible
        frames = self._sort_frames_naturally(frames)

        # Generate (curr, ref) pairs: frame[i] uses frame[i-1] as reference
        # For video sequences: pairs = [(im2,im1), (im3,im2), (im4,im3), ...]
        # Start from index 1 so curr != ref
        pairs = []
        for i in range(1, len(frames)):
            pairs.append((frames[i], frames[i-1]))
            if len(pairs) >= self.frame_pairs_per_sequence:
                break

        return pairs

    def _sort_frames_naturally(self, frames: List[str]) -> List[str]:
        """Sort frames numerically instead of lexicographically."""
        def get_frame_number(path):
            filename = os.path.basename(path)
            # Extract number from filename like "im001.png" or "frame_001.jpg"
            name = os.path.splitext(filename)[0]
            # Try to extract trailing number
            import re
            numbers = re.findall(r'\d+', name)
            if numbers:
                return int(numbers[-1])
            return 0

        return sorted(frames, key=get_frame_number)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a (curr_frame, ref_frame) pair.

        Returns:
            curr_frame: [3, H, W] current frame
            ref_frame: [3, H, W] reference (previous) frame
        """
        curr_path, ref_path = self.pairs[idx]

        # Load and transform images
        curr = Image.open(curr_path).convert('RGB')
        ref = Image.open(ref_path).convert('RGB')

        curr = self.transform(curr)
        ref = self.transform(ref)

        # Reference frame corruption for Phase 3/4
        if self.corruption_sigma > 0:
            noise = torch.randn_like(ref) * self.corruption_sigma
            ref = (ref + noise).clamp(0.0, 1.0)

        return curr, ref


def make_phase1_loader(
    root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    height: int = 256,
    width: int = 448,
) -> DataLoader:
    """
    Create a DataLoader for Phase 1 training.

    Phase 1: No corruption, all frame pairs.

    Args:
        root: Path to video frames
        batch_size: Training batch size
        num_workers: DataLoader workers
        shuffle: Whether to shuffle
        height: Frame height
        width: Frame width

    Returns:
        DataLoader yielding (curr_frame, ref_frame)
    """
    dataset = GenericVideoDataset(
        root=root,
        height=height,
        width=width,
        corruption_sigma=0.0,  # No corruption in Phase 1
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def make_phase3_loader(
    root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    height: int = 256,
    width: int = 448,
    corruption_sigma: float = 0.005,
) -> DataLoader:
    """
    Create a DataLoader for Phase 3/4 training.

    Phase 3/4: With reference frame corruption.

    Args:
        root: Path to video frames
        batch_size: Training batch size
        num_workers: DataLoader workers
        shuffle: Whether to shuffle
        height: Frame height
        width: Frame width
        corruption_sigma: Corruption strength (default 0.005 per design doc)

    Returns:
        DataLoader yielding (curr_frame, ref_frame)
    """
    dataset = GenericVideoDataset(
        root=root,
        height=height,
        width=width,
        corruption_sigma=corruption_sigma,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    # Quick test with synthetic data
    import tempfile
    import numpy as np

    # Create temp directory with fake frames
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 5 fake frames
        for i in range(5):
            arr = np.random.randint(0, 255, (256, 448, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            img.save(os.path.join(tmpdir, f"frame_{i:03d}.png"))

        # Test dataset
        ds = GenericVideoDataset(tmpdir, height=256, width=448)
        print(f"Dataset length: {len(ds)}")
        print(f"Frame pairs: {ds.pairs[:3]}")

        curr, ref = ds[0]
        print(f"curr shape: {curr.shape}")
        print(f"ref shape: {ref.shape}")

        # Test DataLoader
        loader = make_phase1_loader(tmpdir, batch_size=2)
        batch_curr, batch_ref = next(iter(loader))
        print(f"batch_curr shape: {batch_curr.shape}")
        print(f"batch_ref shape: {batch_ref.shape}")

        print("GenericVideoDataset tests passed!")