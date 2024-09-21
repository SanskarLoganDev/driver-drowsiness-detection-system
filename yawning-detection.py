# Headerfiles 
import cv2 
import dlib 
import numpy as np

#Detector and Predictor 
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat" 
predictor = dlib.shape_predictor(PREDICTOR_PATH) 
detector = dlib.get_frontal_face_detector() 

#Facial Landmarks drawing 
def get_landmarks(im): 
    rects = detector(im, 1) 
    if len(rects) > 1: 
        return None 
    if len(rects) == 0: 
        return None 
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

#Annoting the landmarks in the face 
def annotate_landmarks(im, landmarks): 
    im = im.copy() 
    for idx, point in enumerate(landmarks): 
        pos = (point[0, 0], point[0, 1]) 
        cv2.putText(im, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(0, 0, 255)) 
        cv2.circle(im, pos, 3, color=(0, 255, 255)) 
    return im

#Top and Bottom lip points 
def top_lip(landmarks): 
    top_lip_pts = [] 
    for i in range(50,53): 
        top_lip_pts.append(landmarks[i]) 
    for i in range(61,64): 
        top_lip_pts.append(landmarks[i]) 
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts)) 
    top_lip_mean = np.mean(top_lip_pts, axis=0) 
    return int(top_lip_mean[:,1]) 

def bottom_lip(landmarks): 
    bottom_lip_pts = [] 
    for i in range(65,68): 
        bottom_lip_pts.append(landmarks[i]) 
    for i in range(56,59): 
        bottom_lip_pts.append(landmarks[i]) 
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts)) 
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0) 
    return int(bottom_lip_mean[:,1]) 

#The mouth openness measure 
def mouth_open(image): 
    landmarks = get_landmarks(image) 

    if landmarks is None: 
        return image, 0 

    image_with_landmarks = annotate_landmarks(image, landmarks) 
    top_lip_center = top_lip(landmarks) 
    bottom_lip_center = bottom_lip(landmarks) 
    lip_distance = abs(top_lip_center - bottom_lip_center) 
    return image_with_landmarks, lip_distance

#Video capture and main function 
cap = cv2.VideoCapture(0) 
yawns = 0 
yawn_status = False

while True: 
    ret, frame = cap.read() 
    image_landmarks, lip_distance = mouth_open(frame) 

    prev_yawn_status = yawn_status 

    if lip_distance > 25: 
        yawn_status = True 

        cv2.putText(frame, "Subject is Yawning", (50,450), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2) 

        output_text = " Yawn Count: " + str(yawns + 1) 
        cv2.putText(frame, output_text, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2) 

    else: 
        yawn_status = False 

    if prev_yawn_status == True and yawn_status == False: 
        yawns += 1 
    
    cv2.imshow('Live Landmarks', image_landmarks ) 
    cv2.imshow('Yawn Detection', frame ) 

    if cv2.waitKey(1) == 13: #13 is the Enter Key, therefore press the enter key to exit 
        break 
 
cap.release() 
cv2.destroyAllWindows() 