import argparse
import configparser
import csv
import json
from pathlib import Path
from typing import List, Optional


def resolve_dataset_root(data_root: str) -> Path:
    root = Path(data_root)
    candidates = [
        root,
        root / "dataset",
        root / "sportsmot_publish" / "dataset",
    ]
    for candidate in candidates:
        if (candidate / "train").is_dir() or (candidate / "val").is_dir() or (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find extracted SportsMOT dataset folders under {data_root}. "
        "Expected train/val/test or sportsmot_publish/dataset/train."
    )


def resolve_split_root(data_root: str) -> Optional[Path]:
    root = Path(data_root)
    candidates = [
        root / "splits_txt",
        root / "sportsmot_publish" / "splits_txt",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def load_sequences(split_dir: Path, split_file: Optional[Path]) -> List[str]:
    if split_file and split_file.is_file():
        return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir())


def load_seqinfo(seq_dir: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    seqinfo_path = seq_dir / "seqinfo.ini"
    if not seqinfo_path.is_file():
        raise FileNotFoundError(f"Missing seqinfo.ini: {seqinfo_path}")
    cfg.read(seqinfo_path)
    return cfg


def build_split_json(dataset_root: Path, split: str, sequences: List[str]):
    split_dir = dataset_root / split
    out = {
        "images": [],
        "annotations": [],
        "videos": [],
        "categories": [{"id": 1, "name": "pedestrian"}],
    }
    image_cnt = 0
    ann_cnt = 0

    for video_id, seq in enumerate(sequences, start=1):
        seq_dir = split_dir / seq
        if not seq_dir.is_dir():
            continue
        seqinfo = load_seqinfo(seq_dir)
        seq_len = int(seqinfo["Sequence"]["seqLength"])
        im_width = int(seqinfo["Sequence"].get("imWidth", 0))
        im_height = int(seqinfo["Sequence"].get("imHeight", 0))

        out["videos"].append({"id": video_id, "file_name": seq})

        for frame_idx in range(1, seq_len + 1):
            out["images"].append(
                {
                    "file_name": f"{seq}/img1/{frame_idx:06d}.jpg",
                    "id": image_cnt + frame_idx,
                    "frame_id": frame_idx,
                    "prev_image_id": image_cnt + frame_idx - 1 if frame_idx > 1 else -1,
                    "next_image_id": image_cnt + frame_idx + 1 if frame_idx < seq_len else -1,
                    "video_id": video_id,
                    "height": im_height,
                    "width": im_width,
                }
            )

        if split != "test":
            gt_path = seq_dir / "gt" / "gt.txt"
            if not gt_path.is_file():
                raise FileNotFoundError(f"Missing gt file: {gt_path}")
            with gt_path.open(newline="") as fp:
                reader = csv.reader(fp)
                for row in reader:
                    if not row:
                        continue
                    frame_id = int(float(row[0]))
                    track_id = int(float(row[1]))
                    x = float(row[2])
                    y = float(row[3])
                    w = float(row[4])
                    h = float(row[5])
                    conf = float(row[6]) if len(row) > 6 else 1.0
                    ann_cnt += 1
                    out["annotations"].append(
                        {
                            "id": ann_cnt,
                            "category_id": 1,
                            "image_id": image_cnt + frame_id,
                            "track_id": track_id,
                            "bbox": [x, y, w, h],
                            "conf": conf,
                            "iscrowd": 0,
                            "area": w * h,
                        }
                    )

        image_cnt += seq_len

    return out


def write_seqmap(seqmap_path: Path, sequences: List[str]):
    seqmap_path.parent.mkdir(parents=True, exist_ok=True)
    with seqmap_path.open("w", newline="") as fp:
        fp.write("name\n")
        for seq in sequences:
            fp.write(f"{seq}\n")


def main():
    parser = argparse.ArgumentParser("Prepare SportsMOT for BoostTrack")
    parser.add_argument("--data_root", required=True, help="Extracted SportsMOT root or its parent directory")
    parser.add_argument(
        "--seqmap_out",
        default="eval/TrackEval/data/gt/mot_challenge/seqmaps",
        help="Where to write TrackEval seqmaps",
    )
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.data_root)
    split_root = resolve_split_root(args.data_root)
    ann_root = dataset_root / "annotations"
    ann_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue
        split_file = split_root / f"{split}.txt" if split_root else None
        sequences = load_sequences(split_dir, split_file)
        with (ann_root / f"{split}.json").open("w") as fp:
            json.dump(build_split_json(dataset_root, split, sequences), fp)
        write_seqmap(Path(args.seqmap_out) / f"sportsmot-{split}.txt", sequences)
        print(f"Prepared {split}: {len(sequences)} sequences")

    print(f"Dataset root: {dataset_root}")
    print(f"Annotations: {ann_root}")


if __name__ == "__main__":
    main()
