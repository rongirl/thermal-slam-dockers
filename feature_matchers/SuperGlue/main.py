import argparse
from pathlib import Path

import torch
from superglue_handler import SuperGlueDataHandler
from superglue_matcher import SuperGlueFeatureMatcher

torch.set_grad_enabled(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image pair matching and pose evaluation with SuperGlue",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_dir", type=str, help="Path to the directory that contains the images"
    )
    parser.add_argument(
        "--input_pairs", type=str, help="Path to the list of image pairs"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the directory in which the visualization images are written",
    )
    parser.add_argument(
        "--path_to_weights", type=str, help="SuperGlue weights"
    )

    args = parser.parse_args()
    data_handler = SuperGlueDataHandler(args.input_dir, args.output_dir)
    superglue_matcher = SuperGlueFeatureMatcher(args.path_to_weights)
    with open(args.input_pairs, "r") as f:
        pairs = [l.split() for l in f.readlines()]

    for i, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        viz_path = "{}_{}_matches.{}".format(stem0, stem1, "png")
        img0, inp0 = data_handler.read_image(name0)
        img1, inp1 = data_handler.read_image(name1)
        features0 = superglue_matcher.extract_features(inp0)
        features1 = superglue_matcher.extract_features(inp1)
        mkpts0, mkpts1, kpts0, kpts1, color = superglue_matcher.match_features(
            features0, features1
        )
        matches = {
            "image0": img0,
            "image1": img1,
            "kpts0": kpts0,
            "kpts1": kpts1,
            "mkpts0": mkpts0,
            "mkpts1": mkpts1,
            "color": color,
            "path": viz_path,
        }
        data_handler.save_macthes(matches)
