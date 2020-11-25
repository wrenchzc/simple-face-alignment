import cv2
from simple_face_alignment import align_face, ALIGN_METHOD_3POINT, align_faces


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


def test_multi_bbox():
    img = cv2.imread("tests/gxt1.png")
    bbox1 = [313.40, 252.20, 647.00, 627.80]
    landmark1 = [(369.8, 434.59999999999997), (482.59999999999997, 405.79999999999995), (422.59999999999997, 521.0),
                 (457.4, 542.6), (549.8, 504.2)]
    bbox2 = [1218.20, 0, 1578.20, 325.40]
    landmark2 = [(1301.0, 164.6), (1409.0, 116.6), (1377.8, 225.79999999999998), (1392.2, 290.59999999999997),
                 (1497.8, 201.79999999999998)]
    face_info1 = {"landmark": landmark1, "bbox": bbox1}
    face_info2 = {"landmark": landmark2, "bbox": bbox2}
    faces = align_faces(img, [face_info1, face_info2])
    for i, face in enumerate(faces):
        assert (face.shape[:2] == (112, 112))
        cv2.imwrite(f"tests/gxt1_aligned_{i}.jpg", face)
