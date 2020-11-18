import numpy as np
import cv2
import typing
from easydict import EasyDict

STANDARD_FACE_WH = (96, 112)
STANDARD_FACE_WH_N = np.array([STANDARD_FACE_WH], dtype=np.float32)
STANDARD_FACIAL_POINTS = np.array(
    [[30.2946, 51.6963],
     [65.5318, 51.6963],
     [48.0252, 71.7366],
     [33.5493, 92.3655],
     [62.7299, 92.3655]], dtype=np.float32)

LANDMARK_TYPING = typing.List[typing.Tuple]


def _enlarge_bbox(bbox: typing.List[float], enlarge_rate: float, image_size: typing.Tuple) -> typing.List[float]:
    if enlarge_rate <= 1:
        return bbox

    img_h, img_w = image_size
    x, y, r, b = bbox
    w = r - x
    h = b - y
    scale = lambda x: x * (enlarge_rate - 1) / 2
    new_x = x - scale(w) if x - scale(w) > 0 else 0
    new_y = y - scale(h) if y - scale(h) > 0 else 0
    new_r = r + scale(w) if r + scale(w) < img_w else img_w - 1
    new_b = b + scale(h) if b + scale(h) < img_h else img_h - 1
    return [new_x, new_y, new_r, new_b]


# noinspection SpellCheckingInspection
def simple_aligned_face(img: np.ndarray, landmark: LANDMARK_TYPING, bbox: typing.List[float],
                        bbox_enlarge_rate: float = 1
                        ) -> np.ndarray:
    """
    return a aligned face
    :param img: original image, from cv2.imread
    :param landmark:  a list of tuple, 5 point landmark, tuple should be (width, height)
    :param bbox:  a list of numbwe, shoule be  x, y, r, b (x1, y1, x2, y2)
    :param bbox_enlarge_rate:  enlarge bbox as giving rate
    :return: aligned face, opencv format, standard size (96x112)
    """

    enlarged_bbox = _enlarge_bbox(bbox, bbox_enlarge_rate, img.shape[:2])
    x, y, r, b = [int(item) for item in enlarged_bbox]

    landmark_n = np.array(landmark, dtype=np.float32)
    crop_face_image = img[y:b, x:r]
    crop_h, crop_w = crop_face_image.shape[:2]
    rate_n = STANDARD_FACE_WH_N / np.array([crop_w, crop_h], dtype=np.float32)
    landmark_adj = (landmark_n - np.array([x, y], dtype=np.float32)) * rate_n
    crop_face_image = cv2.resize(crop_face_image, dsize=STANDARD_FACE_WH)
    trans_matrix = cv2.getAffineTransform(landmark_adj[:3], STANDARD_FACIAL_POINTS[:3])
    aligned_face = cv2.warpAffine(crop_face_image.copy(), trans_matrix, (STANDARD_FACE_WH[1], STANDARD_FACE_WH[1]))
    return aligned_face
