import cv2
import mediapipe
import scipy.spatial
 
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands
distanceModule = scipy.spatial.distance
 
capture = cv2.VideoCapture(0)
 
frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

line=[]
 
with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:
 
    while (True):
 
        ret, frame = capture.read()
 
        if ret == False:
            continue
 
        frame = cv2.flip(frame, 1)
 
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#        circleColor = (0, 0, 0)
 
        if results.multi_hand_landmarks != None:
 
            normalizedLandmark = results.multi_hand_landmarks[0].landmark[handsModule.HandLandmark.INDEX_FINGER_TIP]
            pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                      normalizedLandmark.y,
                                                                                      frameWidth,
                                                                                      frameHeight)
            normalizedLandmark1 = results.multi_hand_landmarks[0].landmark[handsModule.HandLandmark.MIDDLE_FINGER_TIP]
            pixelCoordinatesLandmark1 = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark1.x,
                                                                                      normalizedLandmark1.y,
                                                                                      frameWidth,
                                                                                      frameHeight)
            
            normalizedLandmark2 = results.multi_hand_landmarks[0].landmark[handsModule.HandLandmark.RING_FINGER_TIP]
            pixelCoordinatesLandmark2 = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark2.x,
                                                                                      normalizedLandmark2.y,
                                                                                      frameWidth,
                                                                                      frameHeight)

            cv2.circle(frame, pixelCoordinatesLandmark, 2, (255,0,0), -1)

            if distanceModule.euclidean(pixelCoordinatesLandmark, pixelCoordinatesLandmark1) > 50:
                line.append(pixelCoordinatesLandmark)
            if distanceModule.euclidean(pixelCoordinatesLandmark, pixelCoordinatesLandmark2) < 70:
                for t in line:
                    if distanceModule.euclidean(pixelCoordinatesLandmark, t) < 10:
                        line.remove(t)

        for i in line:
            cv2.circle(frame, i, 2, (255,0,0), -1)
            
 
        cv2.imshow('Test image', frame)
 
        if cv2.waitKey(1) == 27:
            break
 
cv2.destroyAllWindows()
capture.release()