import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "dancetrack"
DEFAULT_GT_DIR = ROOT / "results" / "gt"


def find_sequence_dirs(root: Path):
    seqs = []
    for path in root.rglob("*"):
        if path.is_dir() and (path / "img1").is_dir() and (path / "seqinfo.ini").is_file():
            seqs.append(path)
    return sorted(seqs)


def extract_split(zip_paths, split_dir: Path):
    split_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in zip_paths:
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing zip: {zip_path}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_root)
            seq_dirs = find_sequence_dirs(tmp_root)
            if not seq_dirs:
                raise RuntimeError(f"No sequence directories found in {zip_path}")
            for seq_dir in seq_dirs:
                dst = split_dir / seq_dir.name
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(seq_dir), str(dst))


def write_seqmap(seqmap_path: Path, seq_names):
    seqmap_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["name\n"]
    lines.extend(f"{name}\n" for name in seq_names)
    seqmap_path.write_text("".join(lines), encoding="utf-8")


def copy_val_gt(data_dir: Path, gt_dir: Path):
    val_root = data_dir / "val"
    out_root = gt_dir / "DANCE-val"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    seq_names = []
    for seq_dir in sorted(val_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        if not (seq_dir / "img1").is_dir():
            continue
        seq_names.append(seq_dir.name)
        dst = out_root / seq_dir.name
        (dst / "gt").mkdir(parents=True, exist_ok=True)
        shutil.copy2(seq_dir / "seqinfo.ini", dst / "seqinfo.ini")
        shutil.copy2(seq_dir / "gt" / "gt.txt", dst / "gt" / "gt.txt")

    write_seqmap(gt_dir / "seqmaps" / "DANCE-val.txt", seq_names)


def write_test_seqmap(data_dir: Path, gt_dir: Path):
    test_root = data_dir / "test"
    if not test_root.exists():
        return
    seq_names = []
    out_root = gt_dir / "DANCE-test"
    out_root.mkdir(parents=True, exist_ok=True)
    for seq_dir in sorted(test_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        if not (seq_dir / "img1").is_dir():
            continue
        seq_names.append(seq_dir.name)
        dst = out_root / seq_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        if (seq_dir / "seqinfo.ini").exists():
            shutil.copy2(seq_dir / "seqinfo.ini", dst / "seqinfo.ini")
    write_seqmap(gt_dir / "seqmaps" / "DANCE-test.txt", seq_names)


def run_convert_script():
    subprocess.run([sys.executable, str(ROOT / "data" / "tools" / "convert_dance_to_coco.py")], check=True, cwd=str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser("Prepare DanceTrack for BoostTrack")
    parser.add_argument("--src_dir", type=str, default="", help="directory containing train1.zip/train2.zip/val.zip/test1.zip/test2.zip")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--gt_dir", type=str, default=str(DEFAULT_GT_DIR))
    parser.add_argument("--skip_extract", action="store_true", help="skip zip extraction and only rebuild annotations/gt")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    gt_dir = Path(args.gt_dir)

    if not args.skip_extract:
        if not args.src_dir:
            raise RuntimeError("--src_dir is required unless --skip_extract is used.")
        src_dir = Path(args.src_dir)
        extract_split([src_dir / "train1.zip", src_dir / "train2.zip"], data_dir / "train")
        extract_split([src_dir / "val.zip"], data_dir / "val")
        extract_split([src_dir / "test1.zip", src_dir / "test2.zip"], data_dir / "test")

    run_convert_script()
    copy_val_gt(data_dir, gt_dir)
    write_test_seqmap(data_dir, gt_dir)

    print("DanceTrack preparation complete.")
    print(f"data_dir: {data_dir}")
    print(f"gt_dir: {gt_dir}")


if __name__ == "__main__":
    main()
