import numpy as np
import cv2
import typing
from numpy.linalg import inv, lstsq
from numpy.linalg import matrix_rank as rank

STANDARD_FACE_WH = (96, 112)
STANDARD_FACE_WH_N = np.array([STANDARD_FACE_WH], dtype=np.float32)
STANDARD_FACIAL_POINTS = np.array(
    [[30.2946, 51.6963],
     [65.5318, 51.6963],
     [48.0252, 71.7366],
     [33.5493, 92.3655],
     [62.7299, 92.3655]], dtype=np.float32)

LANDMARK_TYPING = typing.List[typing.Tuple]

ALIGN_METHOD_5POINT = 1
ALIGN_METHOD_3POINT = 2


def find_non_reflective_similarity(uv, xy, K=2):
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))

    # We know that X * r = U
    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U, rcond=None)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss, sc, 0],
        [tx, ty, 1]
    ])
    T = inv(Tinv)
    T[:, 2] = np.array([0, 0, 1])
    T = T[:, 0:2].T
    return T


def _enlarge_bbox(bbox: typing.List[float], enlarge_rate: float, image_size: typing.Tuple) -> typing.List[float]:
    if enlarge_rate <= 1:
        return bbox

    img_h, img_w = image_size
    x, y, r, b = bbox
    w = r - x
    h = b - y

    def scale(v): return v * (enlarge_rate - 1) / 2

    new_x = x - scale(w) if x - scale(w) > 0 else 0
    new_y = y - scale(h) if y - scale(h) > 0 else 0
    new_r = r + scale(w) if r + scale(w) < img_w else img_w - 1
    new_b = b + scale(h) if b + scale(h) < img_h else img_h - 1
    return [new_x, new_y, new_r, new_b]


# noinspection SpellCheckingInspection
def align_face(img: np.ndarray, landmark: LANDMARK_TYPING, bbox: typing.List[float],
               align_method: int = ALIGN_METHOD_5POINT, enlarge_rate: float = 1
               ) -> np.ndarray:
    """
    return a aligned face
    :param img: original image, from cv2.imread
    :param landmark:  a list of tuple, 5 point landmark, tuple should be (width, height)
                      landmark should be left eye, right eye, nose, mouse left, mouse right
    :param bbox:  a list of numbwe, shoule be  x, y, r, b (x1, y1, x2, y2)
    :param align_method:  align should be "3pointAffine" and "5pointAffine", default is 5pointAffine
    :param enlarge_rate:  enlarge rate for bbox
    :return: aligned face, opencv format, standard size (96x112)
    """

    enlarged_bbox = _enlarge_bbox(bbox, enlarge_rate, img.shape[:2])
    x, y, r, b = [round(item) for item in enlarged_bbox]

    landmark_n = np.array(landmark, dtype=np.float32)
    crop_face_image = img[y:b, x:r]
    crop_h, crop_w = crop_face_image.shape[:2]
    rate_n = STANDARD_FACE_WH_N / np.array([crop_w, crop_h], dtype=np.float32)
    landmark_adj = (landmark_n - np.array([x, y], dtype=np.float32)) * rate_n
    crop_face_image = cv2.resize(crop_face_image, dsize=STANDARD_FACE_WH)
    if align_method == ALIGN_METHOD_3POINT:
        trans_matrix = cv2.getAffineTransform(landmark_adj[:3], STANDARD_FACIAL_POINTS[:3])
    else:
        trans_matrix = find_non_reflective_similarity(landmark_adj, STANDARD_FACIAL_POINTS)

    max_standard_side = max(STANDARD_FACE_WH)
    aligned_face = cv2.warpAffine(crop_face_image.copy(), trans_matrix, (max_standard_side, max_standard_side))
    return aligned_face
