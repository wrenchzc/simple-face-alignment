# simple-face-alignment
a simple face alignment, standard face size is 96x112

example
example image is https://github.com/wrenchzc/simple-face-alignment/blob/main/tests/ty1.jpg

before align


![no align](https://raw.githubusercontent.com/wrenchzc/simple-face-alignment/main/tests/ty1_bbox.jpg)


use 5 point affine


![5 point align](https://raw.githubusercontent.com/wrenchzc/simple-face-alignment/main/tests/ty1_aligned_1.jpg "5 point affine")

use 3 point affine


![3 point align](https://raw.githubusercontent.com/wrenchzc/simple-face-alignment/main/tests/ty1_aligned_3.jpg "3 point affine")




INSTALLATION
############

Currently it is only supported Python3.4 onwards. It can be installed through pip:

.. code:: bash

    $ pip install simple-face-alignment

This implementation requires OpenCV>=4.1 

USAGE
#####

The following example illustrates the ease of use of this package:


.. code:: python

    >>> from simple_face_alignment import align_face, ALIGN_METHOD_3POINT
    >>>
    >>> img = cv2.imread("ty1.jpg")
    >>> # detector result should include a bbox and a landmark
    >>> # bbox is x, y, r, b, landmark is left eye, right eye, nose, mouse left, mouse right
    >>> bbox = [409.89, 230.20, 725.06, 557.34]
    >>> landmark = [(497.8352508544922, 394.51161527633667), (610.6470794677734, 335.02099990844727),
                    (588.0000686645508, 446.7395896911621), (567.606155872345, 543.8972625732422),
                    (643.1047439575195, 437.53148651123047)]
    >>> face_aligned = align_face(img, landmark, bbox)

LICENSE
#######

`MIT License`_.


REFERENCE
=========
[从零开始搭建人脸识别系统（二）：人脸对齐](https://zhuanlan.zhihu.com/p/61343643)
