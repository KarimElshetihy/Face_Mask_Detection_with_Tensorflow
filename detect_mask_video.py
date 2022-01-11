from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from imutils.video.videostream import VideoStream
# import imutils
import time
import numpy as np
import argparse
import cv2
import os


def detect_mask(image, facNet, faceMask):
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    faces = []
    locs = []    
    preds = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args['confidence']:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w-1, endX), min(h-1, endY))

            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224,224))
            face = img_to_array(face)

            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
            
            if len(faces) > 0:
                facesTotal = np.array(faces, dtype='float32')
                preds = maskNet.predict(facesTotal, batch_size=32)
                
    return (locs, preds)


ap = argparse.ArgumentParser()

ap.add_argument('-v', '--video', type=str, default="0", help="path to input image")
ap.add_argument('-f', '--face', type=str, default="Models", help="path to face mask detector model directory")
ap.add_argument('-m', '--model', type=str, default="Models/mask_detector.model", help="path to trained face mask detector model")
ap.add_argument('-c', '--confidence', type=float, default=0.8, help="Minimum Probability to filter Detections")

args = vars(ap.parse_args())


print("[INFO] Loading Face Detector Model...")

prototxtPath = os.path.sep.join([args['face'], 'deploy.prototxt.txt'])
weightsPath = os.path.sep.join([args['face'], "res10_300x300_ssd_iter_140000.caffemodel"])

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)



print("[INFO] Loading Face Mask Detector Model...")

maskNet = load_model(args['model'])

print("[INFO] Starting Video Stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)


while True:
    image = vs.read()
    image = cv2.resize(image, dsize=(800,400))
    
    (locs, preds) = detect_mask(image,  faceNet, maskNet)
    
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        
        label = 'Mask' if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)

        label = '{}: {:.2f}%'.format(label, max(mask, withoutMask)*100)

        cv2.putText(image, label, (startX,startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    cv2.imshow('Output', image)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
        
cv2.destroyAllWindows()
vs.stop()










