from mtcnn.mtcnn import  MTCNN
import cv2

img = cv2.imread('./ronaldo2.jpg')
img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
detector = MTCNN()
box = detector.detect_faces(img_bgr)[0]
(x,y,w,h) = box['box']
face = img[y:y+h, x:x+w]
cv2.imwrite('face4.jpg', face)
img = cv2.rectangle(img, (x,y), (x+w, y+h), 255, 2)
cv2.imshow('Original image', img)
cv2.imshow('Face', face)
cv2.waitKey(0)


