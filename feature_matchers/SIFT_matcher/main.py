import argparse
from pathlib import Path

from SIFT_handler import SIFTDataHandler

from SIFT_matcher import SIFTFeatureMatcher

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
    data_handler = SIFTDataHandler(args.input_dir, args.output_dir)
    sift_matcher = SIFTFeatureMatcher()
    with open(args.input_pairs, "r") as f:
        pairs = [l.split() for l in f.readlines()]

    for i, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        viz_path = "{}_{}_matches.{}".format(stem0, stem1, "png")
        img0 = data_handler.read_image(name0)
        img1 = data_handler.read_image(name1)
        features0 = sift_matcher.extract_features(img0)
        features1 = sift_matcher.extract_features(img1)
        matches = sift_matcher.match_features(features0, features1)
        matches = {
            "image0": img0,
            "image1": img1,
            "kpts0": features0["keypoints"],
            "kpts1": features1["keypoints"],
            "matches": matches,
            "path": viz_path,
        }
        data_handler.save_macthes(matches)
