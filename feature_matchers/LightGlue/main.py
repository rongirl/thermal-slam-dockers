import argparse
from pathlib import Path

import torch
from lightglue_handler import LightGlueDataHandler
from lightglue_matcher import LightGlueFeatureMatcher

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

    args = parser.parse_args()
    data_handler = LightGlueDataHandler(args.input_dir, args.output_dir)
    lightglue_matcher = LightGlueFeatureMatcher()
    with open(args.input_pairs, "r") as f:
        pairs = [l.split() for l in f.readlines()]

    for i, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        viz_path = "{}_{}_matches.{}".format(stem0, stem1, "png")
        img0, inp0 = data_handler.read_image(name0)
        img1, inp1 = data_handler.read_image(name1)
        features0 = lightglue_matcher.extract_features(inp0.to("cpu"))
        features1 = lightglue_matcher.extract_features(inp1.to("cpu"))
        mkpts0, mkpts1 = lightglue_matcher.match_features(features0, features1)
        matches = {
            "image0": img0,
            "image1": img1,
            "mkpts0": mkpts0,
            "mkpts1": mkpts1,
            "path": viz_path,
        }
        data_handler.save_macthes(matches)
