import argparse
from pathlib import Path
import gdown


def main():
    parser = argparse.ArgumentParser(
        description="Download file or folder from Google Drive using gdown."
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Google Drive share URL",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (for file) or output folder path (for folder)",
    )
    parser.add_argument(
        "--folder",
        action="store_true",
        help="Use this flag if the URL is a Google Drive folder link",
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    if args.folder:
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Downloading folder to: {output_path.resolve()}")
        gdown.download_folder(
            url=args.url,
            output=str(output_path),
            quiet=False,
            use_cookies=False,
        )
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading file to: {output_path.resolve()}")
        gdown.download(
            url=args.url,
            output=str(output_path),
            quiet=False,
            use_cookies=False,
        )

    print("Download completed.")


if __name__ == "__main__":
    main()

# Example
# python3 download_from_gdrive_gdown.py --url https://drive.google.com/file/d/1TqyOeXxUy_uEv_Fb3sqmd8XUB9rWJKn9/view?usp=sharing --output data_preprocessed.zip