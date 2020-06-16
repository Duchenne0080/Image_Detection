import numpy as np
import cv2 as cv

CONFIDENCE = 0.5
THRESHOLD = 0.4

net = cv.dnn.readNetFromDarknet('model_data/yolov3-tiny.cfg', 'model_data/yolov3-tiny.weights')

def predict_img(img):
    blobImg = cv.dnn.blobFromImage(img, 1.0/255.0, (416, 416), None, True, False)
    net.setInput(blobImg)
    outInfo = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(outInfo)
    confidences = []
    classIDs = []
    predict_Ary =[]
    for out in layerOutputs:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > CONFIDENCE:
                confidences.append(float(confidence))
                classIDs.append(classID)
    with open('model_data/coco.names', 'rt') as f:
        labels = f.read().rstrip('\n').split('\n')
    if len(classIDs) > 0:
        for index in range(len(classIDs)):
            predict_dic = dict()
            predict_dic['label'] = labels[classIDs[index]]
            predict_dic['accuracy'] = str(confidences[index])
            predict_Ary.append(predict_dic)
    return predict_Ary
if __name__ == '__main__':
    img = cv.imread('model_data/test2.jpg')
    rest_ary = predict_img(img)
    print('Result', rest_ary)
