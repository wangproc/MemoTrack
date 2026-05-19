import argparse
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser("Create a flat zip submission from tracker txt files")
    parser.add_argument("--input_dir", required=True, help="Directory containing MOT txt result files")
    parser.add_argument("--output_zip", required=True, help="Output zip path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No .txt files found in {input_dir}")

    output_zip = Path(args.output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for txt_file in txt_files:
            zf.write(txt_file, arcname=txt_file.name)

    print(f"Created {output_zip} with {len(txt_files)} txt files")


if __name__ == "__main__":
    main()
