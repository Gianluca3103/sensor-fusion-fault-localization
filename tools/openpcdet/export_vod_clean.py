"""Export clean VoD point clouds and GT labels for OpenPCDet CustomDataset."""

from __future__ import annotations

import argparse
import json

from pcdet_integration.vod_dataset import export_vod_custom_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--label-root")
    args = parser.parse_args()
    manifest = export_vod_custom_dataset(
        args.vod_root,
        args.output_root,
        overwrite=args.overwrite,
        label_root=args.label_root,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
