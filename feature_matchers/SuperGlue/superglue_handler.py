from pathlib import Path

import cv2
import numpy as np
import torch

from feature_matchers.common.data_handler import DataHandler


class SuperGlueDataHandler(DataHandler):
    def __init__(
        self,
        input_path: Path,
        output_path: Path,
    ):
        super().__init__(input_path, output_path)
        output_path.mkdir(exist_ok=True)
        self.resize = [640, 480]

    def _resize_image(self, image):
        return cv2.resize(
            image, (self.resize[0], self.resize[1]), interpolation=cv2.INTER_AREA
        )

    def read_image(self, filename):
        path_to_image = Path(self.input_path) / filename
        image = cv2.imread(str(path_to_image), 0)
        if image is None:
            return None, None
        image = self._resize_image(image)
        img_tensor = torch.from_numpy(image / 255.0).float()[None, None]
        return image, img_tensor

    def save_macthes(self, matches):
        margin = 10
        H0, W0 = matches["image0"].shape
        H1, W1 = matches["image1"].shape
        H, W = max(H0, H1), W0 + W1 + margin

        out = 255 * np.ones((H, W), np.uint8)
        out[:H0, :W0] = matches["image0"]
        out[:H1, W0 + margin :] = matches["image1"]
        out = np.stack([out] * 3, -1)
        color = (np.array(matches["color"][:, :3]) * 255).astype(int)[:, ::-1]
        mkpts0, mkpts1 = np.round(matches["mkpts0"]).astype(int), np.round(
            matches["mkpts1"]
        ).astype(int)
        for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
            c = c.tolist()
            cv2.line(
                out,
                (x0, y0),
                (x1 + margin + W0, y1),
                color=c,
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.imwrite(str(Path(self.output_path) / matches["path"]), out)
