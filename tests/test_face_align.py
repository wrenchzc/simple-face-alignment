import cv2
from simple_face_alignment import align_face, ALIGN_METHOD_3POINT


def test_single_bbox():
    img = cv2.imread("tests/ty1.jpg")
    bbox = [409.89, 230.20, 725.06, 557.34]
    landmark = [(497.8352508544922, 394.51161527633667), (610.6470794677734, 335.02099990844727),
                (588.0000686645508, 446.7395896911621), (567.606155872345, 543.8972625732422),
                (643.1047439575195, 437.53148651123047)]

    face_aligned_1 = align_face(img, landmark, bbox)
    cv2.imwrite("tests/ty1_aligned_1.jpg", face_aligned_1)
    assert (face_aligned_1.shape[:2] == (112, 112))
    face_aligned_2 = align_face(img, landmark, bbox, enlarge_rate=1.2)
    cv2.imwrite("tests/ty1_aligned_2.jpg", face_aligned_2)
    assert (face_aligned_2.shape[:2] == (112, 112))
    face_aligned_3 = align_face(img, landmark, bbox, enlarge_rate=1.2, align_method=ALIGN_METHOD_3POINT)
    cv2.imwrite("tests/ty1_aligned_3.jpg", face_aligned_3)
    assert (face_aligned_3.shape[:2] == (112, 112))
