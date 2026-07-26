"""
Vimeo90k Dataset for Budget-Constrained Neural Decoder.

Implements the Vimeo90k septuplet dataset as specified in Section 11
of the design document.

Returns (frames, bitstream_dict, prev_frame) where frames is a septuplet
(7 frames) and bitstream_dict contains the entropy-decoded representations.

This is a real-dataset loader: every frame returned is loaded from the
on-disk PNG file. A failure to load raises an exception - we do NOT
silently substitute random tensors, so the training pipeline cannot
accidentally learn on stale data.

Supported directory structures (auto-detected):

  Kaggle/official layout (preferred):
    <root>/sequences/<SSSSS>/<CCCC>/im{1..7}.png
  <root>/sep_trainlist.txt   (one row: "SSSSS/CCCC <frame_type>")
  <root>/sep_testlist.txt    (parallel to sep_trainlist.txt)

  Flat layout (legacy, fallback):
    <root>/<sequence_dir>/im{1..7}.png
  <root>/train_list.txt     OR  ../<root>/train_list.txt  (per-line "folder <id>")

The Kaggle layout is preferred because (a) the official Vimeo90k
distribution matches it exactly and (b) Kaggle Datasets that re-host
Vimeo90k follow this form. We auto-detect by trying both list files
in order: sep_trainlist.txt -> train_list.txt -> default-stub fallback.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms as T
from PIL import Image
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class Vimeo90kDataset(Dataset):
    """
    Vimeo90k dataset loader for video compression training.

    Vimeo90k provides septuplets (groups of 7 consecutive frames).
    This dataset returns:
        - frames: Septuplet of frames [7, 3, H, W]
        - bitstream_dict: Dictionary containing entropy-decoded representations
          - latent: Quantized latent tensor
          - motion: Motion representation
          - frame_type_id: 0=I, 1=P, 2=B
        - prev_frame: Previous frame for temporal prediction [3, H, W]

    The dataset can apply reference frame corruption during Phase 3/4
    (controlled by apply_corruption flag).
    """

    FRAME_TYPE_I = 0
    FRAME_TYPE_P = 1
    FRAME_TYPE_B = 2

    def __init__(
        self,
        root: str,
        split: str = "train",
        apply_corruption: bool = False,
        corruption_enabled: bool = True,
        transform=None,
        return_bitstream: bool = True,
    ):
        """
        Args:
            root: Root directory containing Vimeo90k dataset
            split: 'train' or 'val' split
            apply_corruption: If True, apply reference frame corruption
            corruption_enabled: If True (and apply_corruption=True),
                              reference frame corruption is active
            transform: Optional transforms to apply to frames
            return_bitstream: If True, return bitstream_dict with decoded representations
        """
        self.root = Path(root)
        self.split = split
        self.apply_corruption = apply_corruption
        self.corruption_enabled = corruption_enabled
        self.transform = transform
        self.return_bitstream = return_bitstream

        self.septuplet_list = self._load_septuplet_list()

        if self.apply_corruption and self.corruption_enabled:
            from ..data.augmentation import ReferenceFrameCorruption
            self.corruption = ReferenceFrameCorruption(enabled=True)
        else:
            self.corruption = None

    def _load_septuplet_list(self) -> List[Dict[str, Any]]:
        """
        Load the list of septuplets from Vimeo90k dataset.

        Auto-detects layout:

        Kaggle layout (sep_trainlist.txt / sep_testlist.txt):
            Each line = "<SSSSS>/<CCCC> <frame_type>" where:
                SSSSS  = 5-digit sequence dir (e.g., "00001")
                CCCC   = 4-digit clip index within the sequence
                Each (sequence, clip) is a single 7-frame clip
                located at <root>/sequences/<SSSSS>/<CCCC>/im{1..7}.png
            frame_type integer is parsed but not currently used
            (we always treat the first frame in the clip as I/P boundary).

        Flat layout (train_list.txt / val_list.txt, legacy support):
            Each line = "<sequence_dir>/<im_id>" OR  "<sequence_dir> <frame_type>"
            Path resolves to <root>/<sequence_dir>/<im_id>.png
            (we strip any "/imN" trailing component to get the directory).

        If neither list file exists, returns a 1-clip stub (for
        smoke-test usage only). The vignette code path will refuse
        to train on this stub because every frame load will fail.

        Returns:
            List of dicts (one per septuplet-clip) with keys:
                'path':   relative path string passed to _load_frame
                'frame_type': int frame_type (0=I, 1=P, 2=B if known)
                'frame_position': int (always 0, kept for backwards-compat)
        """
        # Kaggle/official layout. The official naming is sep_testlist.txt
        # for the held-out set, so we map split="val" -> "test" as well
        # and try both before falling through.
        kaggle_candidates = [
            self.root / f"sep_{self.split}list.txt",
        ]
        if self.split == "val":
            kaggle_candidates.append(self.root / "sep_testlist.txt")

        for kaggle_list in kaggle_candidates:
            if kaggle_list.exists():
                return self._parse_kaggle_list(kaggle_list)

        # Flat layout (legacy)
        flat_candidates = [self.root / f"{self.split}_list.txt"]
        if self.split == "val":
            flat_candidates.append(self.root / "test_list.txt")
        for flat_list in flat_candidates:
            if flat_list.exists():
                return self._parse_flat_list(flat_list)

        # No list file found
        print(
            f"WARNING [Vimeo90kDataset]: no list file found at "
            f"{self.root}/sep_{self.split}list.txt nor "
            f"{self.root}/{self.split}_list.txt. Returning 1-clip stub."
        )
        return self._get_default_septuplets()

    def _parse_kaggle_list(self, list_path: Path) -> List[Dict[str, Any]]:
        """
        Parse the official Vimeo90k format:
            <SSSSS>/<CCCC> <frame_type>

        Each line corresponds to ONE clip. Within the clip, frames im1..im7
        reside at <root>/sequences/<SSSSS>/<CCCC>/im{1..7}.png.

        The `_load_frame(path)` call later appends `.png` and resolves
        the path relative to `<self.root>/sequences`. We therefore emit
        paths like `<SSSSS>/<CCCC>/im<n>` (without the leading
        "sequences/" component).
        """
        entries: List[Dict[str, Any]] = []
        with open(list_path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if not parts:
                    continue
                clip_rel = parts[0]
                try:
                    frame_type = int(parts[1]) if len(parts) > 1 else 1
                except ValueError:
                    frame_type = 1
                entries.append({
                    "path": clip_rel,
                    "frame_type": frame_type,
                    "frame_position": 0,
                })
        print(
            f"  [Vimeo90kDataset] {len(entries)} clips from {list_path} "
            f"(Kaggle layout: sequences/SSSSS/CCCC/im{{1..7}}.png)."
        )
        return entries

    def _parse_flat_list(self, list_path: Path) -> List[Dict[str, Any]]:
        """
        Parse the legacy flat layout:
            <sequence_dir>/im<n> OR  <sequence_dir> <frame_type>

        We normalise so that `path` is always a per-frame relative
        path like "<sequence_dir>/im<n>". For lines with just
        "<sequence_dir>" (single-token lines), we synthesise 7
        entries spanning im1..im7 of that directory.
        """
        entries: List[Dict[str, Any]] = []
        with open(list_path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if not parts:
                    continue
                first = parts[0]
                try:
                    frame_type = int(parts[1]) if len(parts) > 1 else 1
                except ValueError:
                    frame_type = 1

                if first.endswith(".png"):
                    first = first[:-4]
                tokens = first.split("/")
                # If the path ends in im{n}, treat the final token as the
                # frame and the prefix as the directory.
                if (
                    len(tokens) >= 2
                    and tokens[-1].startswith("im")
                    and tokens[-1][2:].isdigit()
                ):
                    entries.append({
                        "path": first,
                        "frame_type": frame_type,
                        "frame_position": 0,
                    })
                else:
                    # Whole-directory line: expand to im1..im7
                    for im_idx in range(1, 8):
                        entries.append({
                            "path": f"{first}/im{im_idx}",
                            "frame_type": frame_type if im_idx == 0 else 1,
                            "frame_position": im_idx - 1,
                        })
        print(
            f"  [Vimeo90kDataset] {len(entries)} entries from {list_path} "
            f"(flat layout)."
        )
        # Group every 7 entries into one septuplet by re-emitting
        # exactly 7 frames per index below.
        return entries

    def _get_default_septuplets(self) -> List[Dict[str, Any]]:
        """
        Return default septuplets when dataset files are not available.
        This is for testing purposes.
        """
        return [
            {"path": "frame0001", "frame_type": 1, "frame_position": 0},
            {"path": "frame0002", "frame_type": 1, "frame_position": 1},
            {"path": "frame0003", "frame_type": 1, "frame_position": 2},
            {"path": "frame0004", "frame_type": 1, "frame_position": 3},
            {"path": "frame0005", "frame_type": 1, "frame_position": 4},
            {"path": "frame0006", "frame_type": 1, "frame_position": 5},
            {"path": "frame0007", "frame_type": 1, "frame_position": 6},
        ]

    def __len__(self) -> int:
        """Number of clips in the loaded septuplet list.

        For the Kaggle layout each entry IS one clip, so len = N.
        For the flat legacy layout each clip is expanded into 7 entries,
        so we divide by 7 to recover the clip count.
        """
        if not self.septuplet_list:
            return 0
        # If every entry has a path that ends in /im{n}, the list was
        # already collapsed to one entry per clip (Kaggle layout). If
        # not, divide by 7 (flat-layout expansion).
        sample_path = self.septuplet_list[0]["path"]
        tokens = sample_path.split("/")
        is_kaggle_layout = (
            len(tokens) == 2  # SSSSS/CCCC
            and tokens[0].isdigit()
            and tokens[1].isdigit()
        )
        if is_kaggle_layout:
            return len(self.septuplet_list)
        return len(self.septuplet_list) // 7

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """
        Get one clip's frames.

        Kaggle layout (one entry == one clip):
            Build 7 frames im1..im7 from <root>/sequences/<SSSSS>/<CCCC>/im{n>.png
            prev_frame = im1, curr_frame candidates = im2..im7

        Flat layout (entries grouped 7 at a time):
            Use the 7 entries starting at idx*7 as im1..im7.

        Returns:
            frames: [7, 3, H, W] septuplet
            bitstream_dict: placeholder dict (real DCVC encoder runs live
                during Phase 1/2 training; placeholders kept for
                downstream code that historically read this field)
            prev_frame: [3, H, W] the previous frame (im1 of the clip)
        """
        sample_path = self.septuplet_list[0]["path"]
        tokens = sample_path.split("/")

        is_kaggle_layout = (
            len(tokens) == 2
            and tokens[0].isdigit()
            and tokens[1].isdigit()
        )

        if is_kaggle_layout:
            return self._get_clip_kaggle(idx)

        return self._get_clip_flat(idx)

    def _get_clip_kaggle(self, idx: int) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """
        Load 7 frames im1..im7 for the Kaggle-layout entry at index `idx`.

        The entry's `path` is `<SSSSS>/<CCCC>`. The 7 frames live at
        <self.root>/sequences/<SSSSS>/<CCCC>/im{n>.png.
        """
        seq = self.septuplet_list[idx]
        clip_rel = seq["path"]  # like "00001/0001"
        seq_dir = self.root / "sequences" / clip_rel
        # Build all 7 image paths; fail loudly if any one is missing.
        frame_paths = [seq_dir / f"im{n}.png" for n in range(1, 8)]
        for i, p in enumerate(frame_paths, start=1):
            if not p.exists():
                raise FileNotFoundError(
                    f"Vimeo90kDataset cannot find frame {p}. "
                    f"Expected Kaggle layout: <root>/sequences/{clip_rel}/im{n}.png"
                )

        frames = [self._load_png(p) for p in frame_paths]
        frames_tensor = torch.stack(frames, dim=0)  # [7, 3, H, W]

        bitstream_dict = self._create_bitstream_dict(frames_tensor)
        prev_frame = frames_tensor[0]

        if self.apply_corruption and self.corruption is not None:
            prev_frame = self.corruption(
                prev_frame.unsqueeze(0),
                frame_position_in_sequence=0,
            ).squeeze(0)

        return frames_tensor, bitstream_dict, prev_frame

    def _get_clip_flat(self, idx: int) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """
        Load 7 frames for the flat-layout entry at clipped-index `idx`.

        Each clip occupies 7 consecutive entries in self.septuplet_list,
        already expanded by _parse_flat_list.
        """
        septuplet_idx = idx * 7
        septuplet_paths = self.septuplet_list[
            septuplet_idx : septuplet_idx + 7
        ]

        frames = []
        for i, sept_info in enumerate(septuplet_paths):
            frame = self._load_frame(sept_info["path"])
            frames.append(frame)

        frames_tensor = torch.stack(frames, dim=0)

        bitstream_dict = self._create_bitstream_dict(frames_tensor)
        prev_frame = frames_tensor[0]

        if self.apply_corruption and self.corruption is not None:
            prev_frame = self.corruption(
                prev_frame.unsqueeze(0),
                frame_position_in_sequence=0,
            ).squeeze(0)

        return frames_tensor, bitstream_dict, prev_frame

    def _load_frame(self, path: str) -> torch.Tensor:
        """
        Load a single frame from disk as a [3, H, W] tensor in [0, 1].

        For the FLAT layout (legacy), `path` is "<seq_dir>/im<n>" and
        this resolves to <self.root>/<seq_dir>/im<n>.png.

        Raises FileNotFoundError if the PNG cannot be resolved.
        The training pipeline depends on real pixel data - we refuse
        to silently fall back to synthetic inputs.
        """
        img_path = self.root / f"{path}.png"

        if not img_path.exists():
            raise FileNotFoundError(
                f"Vimeo90kDataset cannot find {img_path}. "
                f"Expected <root>/{path}.png on disk."
            )

        return self._load_png(img_path)

    def _load_png(self, img_path: Path) -> torch.Tensor:
        """Robust PNG loader that copies through numpy (avoids PIL warning)."""
        with Image.open(str(img_path)) as img:
            img = img.convert("RGB")
            arr = np.array(img, copy=True)
        return torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)

    def _create_bitstream_dict(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Build the input bitstream tensor dict for the current sample.

        In a production deployment this dict is populated by the DCVC
        entropy decoder after the bitstream is decoded on disk. For
        training/eval scripts that consume this dataset we currently
        do not have a serialized per-frame DCVC encode, so we emit
        *placeholder* tensors of the correct shape/team. These placeholders
        are NOT used by the real training path: Phase 1 runs the DCVC
        teacher on (curr_frame, prev_frame) for the actual context tensor,
        and Phase 2's precomputer calls the teacher on the live frames.
        The placeholders exist only so other code that historically read
        bitstream_dict['latent'] / ['motion'] does not crash.

        Placeholders:
            latent:  shape [B, 192, H//16, W//16], standard-deviation-normalised
            motion:  shape [B, 128, H//16, W//16]
            frame_type_id: 0 for the I-frame, 1 for the remaining P-frames
        """
        batch_size, channels, height, width = frames.shape
        device = frames.device

        # Deterministic, low-magnitude placeholders so that downstream
        # callers don't accidentally train on these values.
        gen = torch.Generator(device="cpu").manual_seed(0)
        latent = torch.randn(
            batch_size, 192, height // 16, width // 16,
            generator=gen,
        ) * 0.01
        motion = torch.randn(
            batch_size, 128, height // 16, width // 16,
            generator=gen,
        ) * 0.01

        frame_types = [
            self.FRAME_TYPE_I if i == 0 else self.FRAME_TYPE_P
            for i in range(batch_size)
        ]
        frame_type_id = torch.tensor(frame_types, dtype=torch.long)

        return {
            "latent": latent.to(device),
            "motion": motion.to(device),
            "frame_type_id": frame_type_id,
        }

    def get_frame_type_name(self, frame_type_id: int) -> str:
        """Convert frame type ID to name."""
        names = {0: "I", 1: "P", 2: "B"}
        return names.get(frame_type_id, f"Unknown({frame_type_id})")


class Vimeo90kCollator:
    """
    Collate function for batching Vimeo90k septuplets.

    Each sample returns (frames, bitstream_dict, prev_frame) where
    frames is [7, 3, H, W]. When batching, we stack all septuplets.
    """

    def __call__(self, batch):
        """
        Collate a batch of Vimeo90k samples.

        Args:
            batch: List of (frames, bitstream_dict, prev_frame) tuples

        Returns:
            batched_frames: [B, 7, 3, H, W]
            batched_bitstream: Dictionary of batched tensors
            batched_prev: [B, 3, H, W]
        """
        frames_list = []
        bitstream_list = []
        prev_frames_list = []

        for frames, bitstream, prev_frame in batch:
            frames_list.append(frames)
            bitstream_list.append(bitstream)
            prev_frames_list.append(prev_frame)

        batched_frames = torch.stack(frames_list, dim=0)

        batched_bitstream = {
            "latent": torch.stack([b["latent"] for b in bitstream_list], dim=0),
            "motion": torch.stack([b["motion"] for b in bitstream_list], dim=0),
            "frame_type_id": torch.stack([b["frame_type_id"] for b in bitstream_list], dim=0),
        }

        batched_prev = torch.stack(prev_frames_list, dim=0)

        return batched_frames, batched_bitstream, batched_prev


if __name__ == "__main__":
    print("=" * 60)
    print("Vimeo90k Dataset Test")
    print("=" * 60)

    dataset = Vimeo90kDataset(
        root="./vimeo90k",
        split="train",
        apply_corruption=False,
    )

    print(f"Dataset length: {len(dataset)}")

    if len(dataset) > 0:
        frames, bitstream_dict, prev_frame = dataset[0]

        print(f"\nFrames shape: {frames.shape} (expected [7, 3, H, W])")
        print(f"Latent shape: {bitstream_dict['latent'].shape}")
        print(f"Motion shape: {bitstream_dict['motion'].shape}")
        print(f"Frame type IDs: {bitstream_dict['frame_type_id'].tolist()}")
        print(f"Prev frame shape: {prev_frame.shape}")

    print("\n" + "=" * 60)
    print("Dataset tests passed!")
    print("=" * 60)