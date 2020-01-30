import cv2
cam = cv2.VideoCapture(0)
cv2.namedWindow("trial")
img_counter = 0
# Load the model.
cvNet = cv2.dnn.readNet('face-detection-adas-0001.xml',
                     'face-detection-adas-0001.bin')
# Specify target device.
cvNet.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
# Read an image.


cvLmk = cv2.dnn.readNet('facial-landmarks-35-adas-0002.xml',
                     'facial-landmarks-35-adas-0002.bin')
# Specify target device.
cvLmk.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

while True:
    ret, frame = cam.read()
    rows = frame.shape[0]
    cols = frame.shape[1]
    if not ret:
        break
    k = cv2.waitKey(1)


    cvNet.setInput(cv2.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv2.CV_8U))
    cvOut = cvNet.forward()
    
    cvLmk.setInput(cv2.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv2.CV_8U))
    lmkOut = cvLmk.forward()

    # Draw detected faces on the frame.
    for detection in cvOut.reshape(-1,7):
        confidence = float(detection[2])
        xmin = int(detection[3] * cols)
        ymin = int(detection[4] * rows)
        xmax = int(detection[5] * cols)
        ymax = int(detection[6] * rows)
        if confidence > 0.5 and (xmax - xmin) > 25 :
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(255, 255, 255),thickness = 4)
        
    for i in range(0, lmkOut.shape[1], 2):
        x, y = int(xmin+lmkOut[0][i]*(xmax-xmin)), ymin+int(lmkOut[0][i+1]*(ymax-ymin))
        # Draw Facial key points
        cv2.circle(frame, (x, y), 1, color=(255,255,255),thickness = 4)


    cv2.imshow('frame',frame)

cam.release()
cv2.destroyAllWindows()


