# Headerfiles 
from scipy.spatial import distance
from PIL import Image, ImageTk
from imutils import face_utils
import tkinter as tk
import numpy as np
import pygame
import dlib
import cv2

#Aspect Ratio Calculation 
def eye_aspect_ratio(eye): 
    A = distance.euclidean(eye[1], eye[5]) 
    B = distance.euclidean(eye[2], eye[4]) 
    C = distance.euclidean(eye[0], eye[3]) 
    return((A+B) / (2*C)); 

#Drawing eye points 
def drawPoints(eye, frame): 
    for cent in eye: 
        cv2.circle(frame,tuple(cent), 1, (255,255,255), 1); 
    
#Detection of drowsiness 
def drowsy_detection(frame): 
    global COUNTER; 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); 
    faces = detector(gray, 0) #Detect facial points through detector function 
    for face in faces: 
        shape = predictor(gray, face) 

        shape = face_utils.shape_to_np(shape) 
        leftEye = shape[lStart:lEnd] #Get array of coordinates of leftEye and rightEye 
        rightEye = shape[rStart:rEnd] 

        drawPoints(leftEye, frame);# draw the points of the eye 
        drawPoints(rightEye, frame); 
        l_eye = eye_aspect_ratio(leftEye)#calculation of aspect ratio 
        r_eye = eye_aspect_ratio(rightEye) 
        eye = (l_eye + r_eye) / 2 
        cv2.drawContours(frame, [leftEye], -1, (0, 255, 0), 1)#draw the points around the eye 
        cv2.drawContours(frame, [rightEye], -1, (0, 255, 0), 1) 
        
        #Threshold Calculation 
        if(eye < EYE_THRESHOLD): 
            COUNTER += 1; 
            if COUNTER >= EYE_FRAMES: 
                pygame.mixer.music.play(-1) 
                cv2.putText(frame, "ALERT: Drowsiness detected.", (30,50), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0,0,222), 2) 
        else: 
            pygame.mixer.music.stop() 
            COUNTER = 0 
        eye_str = str(eye); 
        cv2.putText(frame, ("Eye Aspect Ratio:"+eye_str[:4]), (10,h-50), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (222,0,0), 1); 
    return frame; 
#Capturing and displaying the picture 
def video_loop(): 
    ret, frame = cam.read() 
    if ret: 
        frame = drowsy_detection(frame); 

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) 
        current_image = Image.fromarray(cv2image) # convert image for PIL 
        imgtk = ImageTk.PhotoImage(image=current_image) # convert image for tkinter 
        panel.imgtk = imgtk # anchor imgtk so it does not be deleted by garbage-collector 
        panel.config(image=imgtk) # show the image 
    root.after(1, video_loop) # call the same function after 30 milliseconds 
    
#Destroying the windows 
def destructor(): 
    print("[INFO] closing...") 
    root.destroy() 
    cam.release() 
    cv2.destroyAllWindows() 
    
#The main function 
if __name__ == "__main__": 
    EYE_THRESHOLD = 0.20 
    EYE_FRAMES = 15 
    COUNTER = 0; 
    pygame.mixer.init() #Initialize Pygame and load music 
    pygame.mixer.music.load('alarm.mp3') 
    detector = dlib.get_frontal_face_detector()#the detector and predictor functions 
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') 
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye'] 
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye'] 
    cam = cv2.VideoCapture(0) 
    h= 480;w= 640; 
    cam.set(3, h); 
    cam.set(4, w); 
    current_image = None 

    print("[INFO] Starting...") 
    root = tk.Tk() 
    root.geometry("720x740") 
    root.resizable(False, False) 
    root.title("TARP Project") 
    root.protocol('WM_DELETE_WINDOW', destructor) 
    title = tk.Label(root, font = "Helvetica 25 bold", text="Drowsiness Detection").pack(anchor='center', padx='5', pady='5') 
    panel = tk.Label(root) 
    panel.pack(anchor ='center') 
    root.config(cursor="arrow") 
    btn = tk.Button(root, text="Exit!", font = "Helvetica 14 bold", command=destructor) 
    btn.pack(anchor='center', ipadx=5, ipady=5, pady = 5) 
    team = tk.Label(root,font = "Helvetica 16 bold",text="Developer: ").pack(anchor='w', ipadx=5, ipady=2, padx = 10) 
    mate1 = tk.Label(root,font = "Helvetica 12 ",text="Sanskar Vidyarthi").pack(anchor='w', padx=50, ipady=1) 
    video_loop()
    
    root.mainloop()