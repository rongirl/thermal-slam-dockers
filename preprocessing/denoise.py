import argparse
from pathlib import Path

import cv2
import numpy as np


def bandpass_filter(image, lower_border, upper_border):
    F = np.fft.fft2(image)
    Fshift = np.fft.fftshift(F)
    M, N = image.shape
    H = np.zeros((M, N), dtype=np.float32)
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            if lower_border <= D <= upper_border:
                H[u, v] = 1
            else:
                H[u, v] = 0
    Gshift = Fshift * H
    G = np.fft.ifftshift(Gshift)
    return np.abs(np.fft.ifft2(G))


def bilateral_filter(image, diam, sigma_color, sigma_space):
    return cv2.bilateralFilter(image, diam, sigma_color, sigma_space)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filters for denoising images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        help="Path to the directory that contains the images",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, help="Path to the directory in which the results are written"
    )
    parser.add_argument(
        "-f",
        "--filter",
        choices={"bilateral", "bandpass"},
        type=str,
        help="Filter to choose from [bilateral, bandpass]",
    )

    parser.add_argument(
        "--upper",
        type=int,
        default=90,
        help="Lower border of frequency in bandpass filter",
    )

    parser.add_argument(
        "--lower",
        type=int,
        default=0,
        help="Upper border of frequency in bandpass filter",
    )

    parser.add_argument(
        "--d",
        type=int,
        default=9,
        help="Diameter of pixels in bilateral filter",
    )

    parser.add_argument(
        "--color",
        type=int,
        default=350,
        help="SigmaColor in in bilateral filter",
    )

    parser.add_argument(
        "--space",
        type=int,
        default=350,
        help="SigmaSpace in in bilateral filter",
    )
    args = parser.parse_args()
    images = Path(args.input_dir).iterdir()
    for image in images:
        name = image.name
        img = cv2.imread(str(image))
        if args.filter == "bilateral":
            img = bilateral_filter(img, args.d, args.color, args.space)
        else:
            img = bandpass_filter(img, args.lower, args.upper)
        output_path = Path(args.output_dir) / name
        cv2.imwrite(str(output_path), img)
