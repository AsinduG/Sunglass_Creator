# import mediapipe
import cv2
import numpy as np
import HandDetect as hd
import autopy
import cvzone

cap = cv2.VideoCapture(0)
detectHand = hd.HandTrack()
smooth = 12
clocX, clocY = 0, 0
plocX, plocY = 0, 0
w = 240
h = 240
rtr = 100
cap.set(3,w)
cap.set(4,h)
wScrn , hScrn = autopy.screen.size()
print(wScrn,hScrn)
number = 8
number1 = 8


while True:
    rate, frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = detectHand.HandDetection(frame)
    lmList, bbox = detectHand.position(frame)
    fingerList, name = detectHand.fingerUp()

    cv2.rectangle(frame, (20, 20), (200, 120), (0, 255, 0), -1)
    cv2.putText(frame, 'Sunglasses1', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    cv2.rectangle(frame, (20, 120), (200, 220), (255, 255, 0), -1)
    cv2.putText(frame, 'Sunglasses2', (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    cv2.rectangle(frame, (20, 220), (200, 320), (0, 255, 255), -1)
    cv2.putText(frame, 'Sunglasses3', (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    cv2.rectangle(frame, (20, 320), (200, 420), (100, 255, 50), -1)
    cv2.putText(frame, 'Sunglasses4', (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    cv2.rectangle(frame, (20, 420), (200, 520), (250, 255, 50), -1)
    cv2.putText(frame, 'Sunglasses5', (30, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

    #for caps
    cv2.rectangle(frame, (1000, 20), (1200, 120), (0, 255, 0), -1)
    cv2.putText(frame, 'Cap1', (1000, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    cv2.rectangle(frame, (1000, 120), (1200, 220), (255, 255, 0), -1)
    cv2.putText(frame, 'cap2', (1000, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    cv2.rectangle(frame, (1000, 220), (1200, 320), (0, 255, 255), -1)
    cv2.putText(frame, 'Cap3', (1000, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    cv2.rectangle(frame, (1000, 320), (1200, 420), (100, 255, 50), -1)
    cv2.putText(frame, 'Cap4', (1000, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    cv2.rectangle(frame, (1000, 420), (1200, 520), (250, 255, 50), -1)
    cv2.putText(frame, 'Cap5', (1000, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

    if len(fingerList) != 0:
        cv2.rectangle(frame, (rtr*2, rtr*2), (int(wScrn)-rtr*3, int(hScrn)-rtr*3), (255, 255, 0), 2)

    #3. check the finger 2 is up and 2 is down
        if fingerList[1] == 1 and fingerList[2] == 0:
            x3 = np.interp(x1, (75, 640-75), (0, wScrn))
            y3 = np.interp(y1, (75, 480-75), (0, hScrn))
            #if only index finder moving mode

            #smoothing
            clocX = plocX + (x3-plocX)/smooth
            clocY = plocY + (y3 - plocY) / smooth

            autopy.mouse.move(clocX, clocY)
            cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

            plocX, plocY = clocX, clocY

    # 3. check the finger 2 and 3 are up
        if fingerList[1] == 1 and fingerList[2] == 1:
            length, frame, arr = detectHand.fingerDistance(frame)
            if length < 70:
                cv2.circle(frame,(arr[4],arr[5]),15,(0,255,0),cv2.FILLED)
                autopy.mouse.click()

                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

                img = cv2.imread('male1.jpg')
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.2)

                xList = []
                yList = []
                wList = []
                nameList = ['sunglasses_PNG59.png','sunglasses_PNG60.png', 'sunglasses_PNG95.png', 'sunglasses_PNG124.png',
                            'sunglasses_PNG127.png', 'sunglasses_PNG139.png']
                capList = ['cap6.png','cap2.png','cap3.png','cap4.png','cap5.png']
                # number = 0

                #for sunglasses
                if 20 < x1 < 200 and 20 < y1 < 120:
                    number = 0

                if 20 < x1 < 200 and 120 < y1 < 220:
                    number = 1

                if 20 < x1 < 200 and 220 < y1 < 320:
                    number = 2

                if 20 < x1 < 200 and 320 < y1 < 420:
                    number = 3
                if 20 < x1 < 200 and 420 < y1 < 520:
                    number = 4

                 #Caps

                if 1000 < x1 < 1200 and 20 < y1 < 120:
                    number1 = 0

                if 1000 < x1 < 1200 and 120 < y1 <220 :
                    number1 = 1

                if 1000 < x1 < 1200 and 220 < y1 <320 :
                    number1 = 2

                if 1000 < x1 < 1200 and 320 < y1 <420 :
                    number1 = 3
                if 1000 < x1 < 1200 and 420 < y1 <520 :
                    number1 = 4



                if 1000 < x1 < 1200 and 520 < y1 <620 :
                    number1 = 8
                    number = 8

                if number < 5:


                    # overlayCap = cv2.imread(f'{capList[number1]}', cv2.IMREAD_UNCHANGED)

                    overlayimg = cv2.imread(f'{nameList[number]}', cv2.IMREAD_UNCHANGED)

                    eyes = eye_cascade.detectMultiScale(gray, 1.2)

                    for x1, y1, w1, h1 in eyes:
                        xList.append(x1)
                        yList.append(y1)
                        wList.append(w1)
                    wMin = min(wList)
                    xMin = min(xList)
                    yMin = min(yList)

                    for (x, y, w, h) in faces:
                        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                        overlayimg_resize = cv2.resize(overlayimg, (int(w1 * 4), int(h1 * 1.4)))
                        # overlayCap_resize = cv2.resize(overlayCap, (int(w*0.9), int(h * 0.6)))
                        img = cvzone.overlayPNG(img, overlayimg_resize, [xMin - int(wMin / 1.2), y + int(h / 3.4)]) #- int(wMin / 2)
                        # img = cvzone.overlayPNG(img, overlayCap_resize, [x+int(w1/4), y -int(h/4.5)])

                        # frame = cvzone.overlayPNG(frame, img)

                if number1 < 5:


                    overlayCap = cv2.imread(f'{capList[number1]}', cv2.IMREAD_UNCHANGED)


                    eyes = eye_cascade.detectMultiScale(gray, 1.2)

                    for x1, y1, w1, h1 in eyes:
                        xList.append(x1)
                        yList.append(y1)
                        wList.append(w1)
                    wMin = min(wList)
                    xMin = min(xList)
                    yMin = min(yList)

                    for (x, y, w, h) in faces:
                        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                        # overlayimg_resize = cv2.resize(overlayimg, (int(w1 * 4), int(h1 * 1.4)))
                        overlayCap_resize = cv2.resize(overlayCap, (int(w), int(h * 0.8)))
                        # img = cvzone.overlayPNG(img, overlayimg_resize, [xMin - int(wMin / 1.7), y + int(h / 3.4)])
                        img = cvzone.overlayPNG(img, overlayCap_resize, [x-5, y -int(h/2)])

                        # frame = cvzone.overlayPNG(frame, img)

                cv2.imshow('img', img)


    if name == 1:
        word = 'Right Hand'
    else:
        word = 'Left Hand'
    cv2.putText(frame,word,(50,50),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2, (200,20,20),2)
    # print(fingerList)
    imS = cv2.resize(frame, (540, 540))
    cv2.imshow('image',imS)
    if cv2.waitKey(1) == ord('q'):
        break

