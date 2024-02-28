from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO('front.pt')


# img = Image.open('front.jpg')
img = cv2.imread("front.jpg")
print(type(img))

results = model(img)

print(results)
