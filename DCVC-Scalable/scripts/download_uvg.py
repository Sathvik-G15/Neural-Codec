"""
Download the official UVG dataset from Tampere University.

This script downloads 8 video sequences (1920x1080, 120fps) from the
official Ultra Video Group dataset at https://ultravideo.cs.tut.fi/

Sequences: Beauty, Bosphorus, HoneyBee, Jockey, ReadySetGo, ShakeNDry, YachtRide, Lips

Usage:
    python scripts/download_uvg.py --output data/uvg
"""

import os
import sys
import argparse
import subprocess
import ssl
from pathlib import Path

UVG_SEQUENCES = {
    'Beauty':      'https://ultravideo.cs.tut.fi/video/Beauty_1920x1080_120fps_420_8bit_YUV_RAW.7z',
    'Bosphorus':   'https://ultravideo.cs.tut.fi/video/Bosphorus_1920x1080_120fps_420_8bit_YUV_RAW.7z',
    'HoneyBee':    'https://ultravideo.cs.tut.fi/video/HoneyBee_1920x1080_120fps_420_8bit_YUV_RAW.7z',
    'Jockey':      'https://ultravideo.cs.tut.fi/video/Jockey_1920x1080_120fps_420_8bit_YUV_RAW.7z',
    'ReadySetGo':  'https://ultravideo.cs.tut.fi/video/ReadySetGo_1920x1080_120fps_420_8bit_YUV_RAW.7z',
    'ShakeNDry':   'https://ultravideo.cs.tut.fi/video/ShakeNDry_1920x1080_120fps_420_8bit_YUV_RAW.7z',
    'YachtRide':   'https://ultravideo.cs.tut.fi/video/YachtRide_1920x1080_120fps_420_8bit_YUV_RAW.7z',
    'Lips':        'https://ultravideo.cs.tut.fi/video/Lips_1920x1080_120fps_420_8bit_YUV_RAW.7z',
}

UVG_FRAMES = {
    'Beauty': 600, 'Bosphorus': 600, 'HoneyBee': 600, 'Jockey': 600,
    'ReadySetGo': 600, 'ShakeNDry': 300, 'YachtRide': 600, 'Lips': 600,
}


def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_curl():
    try:
        subprocess.run(['curl', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_with_curl(url, output_path, name):
    print(f"  Downloading {name} with curl...")
    try:
        result = subprocess.run(
            ['curl', '-L', '-o', str(output_path), '-w', '%{http_code}', '--retry', '3', '--retry-delay', '5', '--connect-timeout', '30', '--max-time', '7200', url],
            capture_output=True, text=True, timeout=7500
        )
        if result.returncode == 0 and result.stdout.strip().endswith('200'):
            return True
        print(f"    curl failed (HTTP {result.stdout.strip()}), trying Python fallback...")
        return False
    except Exception as e:
        print(f"    curl error: {e}, trying Python fallback...")
        return False


def download_with_python(url, output_path, name):
    import urllib.request
    print(f"  Downloading {name} with Python urllib...")
    ssl_context = ssl._create_unverified_context()

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 // total_size)
            mb = downloaded / 1e6
            total_mb = total_size / 1e6
            print(f"\r    {mb:.1f}/{total_mb:.1f} MB ({percent}%)", end='', flush=True)

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
        print()
        return True
    except Exception as e:
        print(f"\n    Python download error: {e}")
        return False


def download_archive(url, output_path, name):
    if output_path.exists() and output_path.stat().st_size > 1000:
        print(f"  Archive already exists ({output_path.stat().st_size / 1e6:.1f} MB), skipping download")
        return True

    if check_curl():
        return download_with_curl(url, output_path, name)
    return download_with_python(url, output_path, name)


def extract_7z(archive_path, output_dir):
    import py7zr
    print(f"  Extracting {archive_path.name}...")
    try:
        with py7zr.SevenZipFile(archive_path, mode='r') as z:
            z.extractall(path=output_dir)
        return True
    except Exception as e:
        print(f"  Extraction error: {e}")
        return False


def find_yuv_file(seq_dir):
    for ext in ['.yuv', '.raw', '.y4m']:
        files = list(seq_dir.glob(f'*{ext}')) + list(seq_dir.glob(f'*{ext.upper()}'))
        if files:
            return files[0]
    return None


def convert_yuv_to_png(yuv_path, seq_dir, name, expected_frames):
    print(f"  Converting YUV to PNG frames...")
    w, h = 1920, 1080
    yuv_base = seq_dir / name

    if yuv_base.with_suffix('.yuv').exists() or yuv_base.with_suffix('.YUV').exists():
        yuv_file = yuv_base.with_suffix('.yuv') if yuv_base.with_suffix('.yuv').exists() else yuv_base.with_suffix('.YUV')
    else:
        yuv_file = yuv_path

    output_pattern = seq_dir / 'frame_%04d.png'

    ffmpeg_cmd = [
        'ffmpeg', '-f', 'rawvideo', '-pix_fmt', 'yuv420p',
        '-s', f'{w}x{h}', '-r', '120',
        '-i', str(yuv_file),
        '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',
        '-frames:v', str(expected_frames),
        '-q:v', '2',
        '-y', str(output_pattern)
    ]

    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"    ffmpeg error: {result.stderr[:500]}")
            return False

        frames = list(seq_dir.glob('frame_*.png'))
        print(f"    Extracted {len(frames)} frames")
        return len(frames) >= expected_frames * 0.9

    except subprocess.TimeoutExpired:
        print(f"    ffmpeg timed out")
        return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


def cleanup_raw_files(seq_dir, yuv_path):
    for ext in ['.yuv', '.YUV', '.raw', '.RAW', '.y4m', '.Y4M']:
        files = list(seq_dir.glob(f'*{ext}'))
        for f in files:
            try:
                f.unlink()
            except Exception:
                pass


def download_and_process_sequence(name, url, output_dir, force=False, resume=False):
    output_dir = Path(output_dir)
    seq_dir = output_dir / name

    frames = list(seq_dir.glob('frame_*.png'))
    if frames and not force:
        if len(frames) >= UVG_FRAMES[name] * 0.9:
            print(f"  {name}: Already downloaded ({len(frames)} frames)")
            return True
        elif resume:
            print(f"  {name}: Found {len(frames)} frames, resume not implemented — skipping (use --force to re-download)")
            return True

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f'{name}.7z'

    if not download_archive(url, archive_path, name):
        if archive_path.exists():
            archive_path.unlink()
        return False

    seq_dir.mkdir(parents=True, exist_ok=True)
    if not extract_7z(archive_path, seq_dir):
        return False

    yuv_path = find_yuv_file(seq_dir)
    if yuv_path is None:
        print(f"  Error: Could not find YUV file after extraction")
        return False

    if not convert_yuv_to_png(yuv_path, seq_dir, name, UVG_FRAMES[name]):
        return False

    cleanup_raw_files(seq_dir, yuv_path)

    if archive_path.exists():
        archive_path.unlink()

    final_frames = list(seq_dir.glob('frame_*.png'))
    print(f"  {name}: Complete ({len(final_frames)} frames)")
    return len(final_frames) >= UVG_FRAMES[name] * 0.9


def list_available(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print("No sequences found")
        return

    sequences = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir():
            frames = sorted(d.glob('frame_*.png'))
            if frames:
                sequences.append((d.name, len(frames)))

    if sequences:
        print("Available sequences:")
        for name, count in sequences:
            print(f"  {name}: {count} frames")
    else:
        print("No sequences found")


def main():
    parser = argparse.ArgumentParser(description='Download UVG dataset from Tampere University')
    parser.add_argument('--output', '-o', type=str, default='data/uvg',
                        help='Output directory')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List already-downloaded sequences')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force re-download even if exists')
    parser.add_argument('--sequence', '-s', type=str, default=None,
                        help='Download specific sequence')
    args = parser.parse_args()

    if not check_ffmpeg():
        print("ERROR: ffmpeg is not installed or not in PATH")
        print("Install ffmpeg: https://ffmpeg.org/download.html")
        print("On Windows: winget install ffmpeg   OR   choco install ffmpeg")
        sys.exit(1)

    if args.list:
        list_available(args.output)
        return

    if args.sequence:
        if args.sequence not in UVG_SEQUENCES:
            print(f"Unknown sequence: {args.sequence}")
            print(f"Available: {', '.join(UVG_SEQUENCES.keys())}")
            sys.exit(1)
        sequences_to_download = [(args.sequence, UVG_SEQUENCES[args.sequence])]
    else:
        sequences_to_download = list(UVG_SEQUENCES.items())

    total_size = sum([
        0.86, 0.63, 0.84, 0.72, 0.77, 0.43, 0.67, 0.84
    ])

    print("=" * 70)
    print("UVG DATASET DOWNLOAD (Official Tampere University)")
    print("=" * 70)
    print(f"Output directory: {args.output}")
    print(f"Sequences: {len(sequences_to_download)}")
    print(f"Est. compressed size: {total_size:.1f} GB")
    print(f"ffmpeg: OK")
    print("=" * 70)

    success = 0
    for i, (name, url) in enumerate(sequences_to_download):
        print(f"\n[{i+1}/{len(sequences_to_download)}] {name}")
        print("-" * 50)
        if download_and_process_sequence(name, url, args.output, force=args.force):
            success += 1
        else:
            print(f"  FAILED: {name}")

    print("\n" + "=" * 70)
    print(f"RESULT: {success}/{len(sequences_to_download)} sequences downloaded")
    print("=" * 70)

    if success == len(sequences_to_download):
        print("\nAll sequences downloaded successfully!")
        list_available(args.output)
    else:
        print(f"\n{len(sequences_to_download) - success} sequences failed.")
        print("Run again with same arguments to resume.")


if __name__ == '__main__':
    main()