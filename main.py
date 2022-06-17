import cv2
import numpy as np
import os
import pyautogui
from imutils import face_utils
from scipy.spatial import distance as dist
import dlib
from datetime import datetime

cascPath = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(cascPath)
cascPath2 = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(cascPath2)
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)
sumarrx = [0]
sumarry = [0]
opvalue = 10
left_eye_landmarks = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
blink_rate = 0.24
blinkTime = []
optimeMode = True
optimEdges = []
GlobalEyeCoor = (0, 0)
MinMs = 500
MaxMs = 2000


def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
    return frame


def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]
    return left_eye, right_eye


def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    eyebrow_w = int(width / 6)
    img = img[eyebrow_h:height, eyebrow_w:width - eyebrow_w]  # cut eyebrows out (15 px)
    return img


def blob_process(img, threshold, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)
    return keypoints


def nothing(x):
    pass


def averageCorner(kps):
    sumx = 0
    sumy = 0
    count = len(kps)
    if not kps.any():
        count = 1
    for e in kps:
        sumx += e[0]
        sumy += e[1]
    return sumx / count, sumy / count


def optimus(coor):
    if coor[0] != 0 and coor[1] != 0:
        if len(sumarrx) < opvalue:
            sumarrx.append(coor[0])
        else:
            sumarrx.pop(0)
            sumarrx.append(coor[0])
        if len(sumarry) < opvalue:
            sumarry.append(coor[1])
        else:
            sumarry.pop(0)
            sumarry.append(coor[1])
    return sum(sumarrx) / len(sumarrx), sum(sumarry) / len(sumarry)


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def blinkFunc(time_now, minBlinkSl, maxBlinkSl):
    if len(blinkTime) == 0:
        blinkTime.append(time_now)
        # print("Blink : ",len(blinkTime))
    else:
        if time_now - blinkTime[len(blinkTime) - 1] > minBlinkSl:
            if time_now - blinkTime[len(blinkTime) - 1] < maxBlinkSl:
                blinkTime.append(time_now)
                # print("Blink : ",len(blinkTime))
            else:
                blinkTime.clear()
                # print("Blink Memory cleaned.")
        elif time_now - blinkTime[-1] < 0:
            blinkTime.clear()
            # print("Blink Memory cleaned.")


def runEvent(nowSec, maxBlinkSl, mode, eyeCenter):
    BlinkCount = len(blinkTime)
    if BlinkCount != 0:
        if nowSec - blinkTime[len(blinkTime) - 1] > maxBlinkSl:
            blinkTime.clear()
            if mode:
                if 1 < BlinkCount < 5:
                    if len(optimEdges) < 3:
                        optimEdges.append(eyeCenter)
                        print(len(optimEdges), ". Coordinate was saved. Coordinate : ", eyeCenter)
                    elif len(optimEdges) == 3:
                        optimEdges.append(eyeCenter)
                        mode = False
                        print(len(optimEdges), ". Coordinate was saved. Coordinate : ", eyeCenter)
                        print("Edges : ", optimEdges)
                    else:
                        mode = False
                elif BlinkCount == 5:
                    print("Start Optimization")
                    optimEdges.clear()
                    mode = True
            else:
                if BlinkCount == 2:
                    print("Left Click")
                    pyautogui.click()
                elif BlinkCount == 3 and not mode:
                    print("Right Click")
                    pyautogui.click(button='right')
                elif BlinkCount == 4 and not mode:
                    print("Double Left Click")
                    pyautogui.doubleClick()
                elif BlinkCount == 5:
                    print("Start Optimization")
                    optimEdges.clear()
                    mode = True
    return mode


def getScreenCoor(screen):
    if len(optimEdges) == 4:
        w1 = optimEdges[1][0] - optimEdges[0][0]
        w2 = optimEdges[3][0] - optimEdges[2][0]
        w = (w1 + w2) / 2
        h1 = optimEdges[1][1] - optimEdges[0][1]
        h2 = optimEdges[3][1] - optimEdges[2][1]
        h = (h1 + h2) / 2
        mulx = screen[0] / w
        muly = screen[1] / h
        originX = (optimEdges[0][0] + optimEdges[2][0]) / 2
        originY = (optimEdges[0][1] + optimEdges[1][1]) / 2
        newX = GlobalEyeCoor[0] - originX
        newY = GlobalEyeCoor[1] - originY
        if newX * mulx < 0:
            newX = 0
        elif newX * mulx > screen[0]:
            newX = screen[0] / mulx
        if newY * muly < 0:
            newY = 0
        elif newY > screen[1]:
            newY = screen[1] / muly
        return newX * mulx, newY * muly
    else:
        print("Error -1")


def main():
    global GlobalEyeCoor, optimeMode
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    ScreenC = pyautogui.size()
    print(ScreenC)
    detectorBlink = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
    cv2.setTrackbarPos('threshold', 'image', 18)
    while True:
        _, frame = cap.read()
        face_frame = detect_faces(frame, face_cascade)
        if face_frame is not None:
            eyes = detect_eyes(face_frame, eye_cascade)
            for eye in eyes:
                if eye is not None:
                    eye = cut_eyebrows(eye)
                    threshold = cv2.getTrackbarPos('threshold', 'image')
                    keypoint = blob_process(eye, threshold, detector)
                    e = cv2.drawKeypoints(eye, keypoint, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    kps = np.array([k.pt for k in keypoint])
                    notOptimized = averageCorner(kps)
                    GlobalEyeCoor = optimus(averageCorner(kps))
                    if not optimeMode:
                        koor = getScreenCoor(ScreenC)
                        pyautogui.FAILSAFE = False
                        x, y = pyautogui.position()
                        cv2.putText(frame, 'Blink Events : ', (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, '2 = left click, 3 = right click', (0, 430),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, '4 = double left click, 5 = Optimize Mode', (0, 460),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        print("X : ", int(koor[0]), "Y : ", int(koor[1]))
                        if int(notOptimized[0]) != 0 and int(notOptimized[1]) != 0:
                            pyautogui.moveTo(int(koor[0]), int(koor[1]), 0.02)
                        elif int(notOptimized[1]) != 0:
                            pyautogui.moveTo(x, int(koor[1]), 0.02)
                        elif int(notOptimized[0]) != 0:
                            pyautogui.moveTo(int(koor[0]), y, 0.02)
                    else:
                        text = " "
                        if len(optimEdges) == 0:
                            text = "look at the top left of the monitor"
                        elif len(optimEdges) == 1:
                            text = "look at the top right of the monitor"
                        elif len(optimEdges) == 2:
                            text = "look at the bottom left of the monitor"
                        elif len(optimEdges) == 3:
                            text = "look at the bottom right of the monitor"
                        cv2.putText(frame, text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, "and blink twice.", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                                    cv2.LINE_AA)
                        cv2.putText(frame, 'Optimize Mode', (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                    cv2.LINE_AA)
        # -------- Blink Detect Area -----------------------------------
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces, _, _ = detectorBlink.run(gray_frame, 0, 0.0)
        for face in faces:
            landmarks = predictor(gray_frame, face)
            shape = face_utils.shape_to_np(landmarks)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            ear = (leftEAR + rightEAR) / 2
            dt = datetime.now()
            nowSec = int(dt.microsecond / 1000) + dt.second * 1000
            optimeMode = runEvent(nowSec, MaxMs, optimeMode, GlobalEyeCoor)
            if ear < blink_rate:
                blinkFunc(nowSec, MinMs, MaxMs)
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
