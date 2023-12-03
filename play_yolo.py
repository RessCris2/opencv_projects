import cv2
import numpy as np

# net = cv2.dnn.readNetFromDarknet('yolov3.cfg', )
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

weights = 'yolov3.weights'
config = 'yolov3.cfg'

ln = net.getLayerNames()
print(len(ln), ln)

image_path = "oic.jpg"
# read input image
image = cv2.imread(image_path)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

# read class names from text file
classes = None
with open(classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes 
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet(weights, config)

# create input blob 
blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

# set input blob for the network
net.setInput(blob)