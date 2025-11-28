from ultralytics import YOLO
import cv2

yolo = YOLO('runs/segment/train/weights/best.pt')
# results = yolo.predict('data/domain_a/test/images/img008_20201027_Kaizu_TCA422_515x515.png', conf=0.7)
results = yolo.predict('data/domain_b/test/images/20220824_WK_abends_1_frame01610.jpg', conf=0.5)
cv2.imshow('test', results[0].plot())

cv2.waitKey(0)

