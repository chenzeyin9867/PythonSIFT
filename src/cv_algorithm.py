import os
import cv2


img_dir = '../input_image/parrington'
names = os.listdir(img_dir)

images = []
for name in names:
    img_path = os.path.join(img_dir, name)
    image = cv2.imread(img_path)
    images.append(image)

stitcher = cv2.Stitcher_create()
images.pop(0)
status, stitched = stitcher.stitch(images)

if status == 0:
    # cv2.imwrite('pictures/stitch.jpg', stitched)
    cv2.imshow('1',stitched)
    cv2.waitKey(0)
