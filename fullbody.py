import cv2
import mediapipe as mp
from cam_ports import list_ports
import numpy as np
import networkx as nx
import glob
import re
import tkinter as tk
from tkinter import Button, Scale, IntVar
from threading import Thread
from animated_sprite import AnimatedSprite
from replace import replace
mp_drawing = mp.solutions.drawing_utils
drawer = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.1,min_tracking_confidence=0.1)

# For webcam input:
camPort = [-1]
''''''
workingPorts = list_ports()[1]
window = tk.Tk()
window.title("Select a Camera")

def setCamPort(n):
    camPort[0] = n
    window.destroy()
for port in workingPorts:
    B = Button(window, text = "Cam Port ["+str(port)+"]", command = lambda: setCamPort(port))
    B.pack()
window.mainloop()

#camPort = [2]
print(camPort)
if camPort[0] == -1:
    exit()

cap = cv2.VideoCapture(camPort[0])


def graphSim(graph1, graph2, minimum_energy = 0.999999999999999999999999999999999999999999):
    
    def select_k(spectrum):
        running_total = 0.0
        total = sum(spectrum)
        if total == 0.0:
            return len(spectrum)
        for i in range(len(spectrum)):
            running_total += spectrum[i]
            if running_total / total >= minimum_energy:
                return i + 1
        return len(spectrum)

    laplacian1 = nx.spectrum.laplacian_spectrum(graph1)
    laplacian2 = nx.spectrum.laplacian_spectrum(graph2)
    
    k1 = select_k(laplacian1)
    k2 = select_k(laplacian2)
    k = max(k1, k2)
    
    similarity = sum((laplacian1[:k] - laplacian2[:k])**2)
    
    return similarity

def bod2Graph(bod_result, scale = 100):
    landmarks = bod_result.landmark
    G = nx.Graph()
    G.add_nodes_from(list(range(21)))
    
    
    
    
    struct_dict = {
      15:{11,31,0},
      19:{11,0},
      16:{12,32,0},
      20:{12,0},
      14:{24,32,13,0},
      13:{23,31,14,0},
      28:{27,0},
      27:{28,0},
      20:{32,0},
      19:{31,0},
      32:{20,0},
      31:{19,0},
      30:{29,0},
      29:{30,0},


    }
    
    
    
    adjecentcy_list = []
    a,b = (12,23)
    reference_vector = np.array([landmarks[a].x,landmarks[a].y,landmarks[a].z])-np.array([landmarks[b].x,landmarks[b].y,landmarks[b].z])
    size_factor = np.linalg.norm(reference_vector)
    
    
    for origin in struct_dict:
        or_landmark = landmarks[origin]
        or_ary = np.array([or_landmark.x,or_landmark.y,or_landmark.z])*scale/size_factor
        for target in struct_dict[origin]:
            tar_landmark = landmarks[target]
            tar_ary = np.array([tar_landmark.x,tar_landmark.y,tar_landmark.z])*scale/size_factor
            #weight is the magnitude of the vector(obtained by subtracting the two positional vectors)
            
            adjecentcy_list.append((origin,target,np.linalg.norm(tar_ary-or_ary)))
    G.add_weighted_edges_from(adjecentcy_list)
    
    return G


filenames = [img for img in glob.glob("poses/*.jpg")]

filenames.sort() # ADD THIS LINE

ref_images = {}
for img in filenames:
    n= cv2.imread(img)
    print(img)
    expr = re.compile("(?<=poses[\/\\\])[^.]*")
    name = expr.search(img).group(0)
    ref_images[name] = n
    print(name)



ref_pGs = {}

for ref_img in ref_images:
    ref_height, ref_width, ref_channels = ref_images[ref_img].shape 
    rgb_ref_img = cv2.cvtColor(ref_images[ref_img],cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_ref_img)
    
    
    #print("image")
    if results.pose_landmarks:
        
        #drawer.draw_landmarks(rgb_ref_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        #cv2.imshow('stuff',cv2.resize(rgb_ref_img,dsize=(640,480)))
        #cv2.waitKey(0)
        ref_pGs[ref_img] = bod2Graph(results.pose_landmarks)

#exit()
shapes = [0,0,0]

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    ix,iy,iz = image.shape
    shapes[0] = iy
    shapes[1] = ix
    shapes[2] = iz
    break



ix,iy,iz = shapes
##Settings Window##
setting = tk.Tk()

#Setting Values
spriteSizeFac = IntVar(setting,100)
xtest = IntVar()
ytest = IntVar()
depth = False
currentRec = tk.StringVar()

lb = tk.Label(setting, text = "X Displacement:")
lb.pack()
w1 = Scale(setting, from_=-ix, to=ix,orient=tk.HORIZONTAL,variable=xtest)
w1.pack()
w1.set(0)
lb = tk.Label(setting, text = "Y Displacement:")
lb.pack()
w2 = Scale(setting, from_=-iy, to=iy,variable=ytest)
w2.pack()
w2.set(0)
lb = tk.Label(setting, text = "Size Factor:")
lb.pack()
w3 = Scale(setting, from_=0, to=600,variable=spriteSizeFac)
w3.pack()
lb = tk.Label(setting, text = "Current Recongition: ")
lb.pack()
rec = tk.Label( setting, textvariable=currentRec, relief=tk.RAISED )
rec.pack()
#sprites
bg = AnimatedSprite('sprites/bg/*.png',fps=10,resize=(1920,1080))
flower = AnimatedSprite('sprites/flower/*.png',fps=7,resize=(600,600))
star = AnimatedSprite('sprites/star/*.png',fps=8,resize=(600,600))
heart = AnimatedSprite('sprites/heart/*.png',fps=10,resize=(600,600))

bgy, bgx, bgz = bg.out().shape
bg.start()
flower.start()
star.start()
heart.start()
spriteBase = {
    "omg": flower,
    "star": star,
    "heart": heart
}



def cvMainLoop():
    with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            canvas = bg.out();
          
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
      
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
              image,
              results.pose_landmarks,
              mp_pose.POSE_CONNECTIONS,
              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            #cv2.circle(image,(xtest.get(),ytest.get()), spriteSizeFac.get(), (0,0,255), -1)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                wristMarks = landmarks[0]
                if wristMarks:
                    cur_pG = bod2Graph(results.pose_landmarks)
                    sim_comp = [(int(graphSim(ref_pGs[ref_pG],cur_pG)),ref_pG) for ref_pG in ref_pGs]
                    sim_comp.sort()
                    
                    #cv2.putText(image,str(str(sim_comp[0][1])+' '+(str(sim_comp[0][0])[::-1][3:][::-1])), (int(wristMarks.x*ix)+20,int(wristMarks.y*iy)), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),2)
                    #cv2.putText(image,str(str(sim_comp[1][1])+' '+(str(sim_comp[1][0])[::-1][3:][::-1])), (int(wristMarks.x*ix)+20,int(wristMarks.y*iy)+30), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),2)
                    #cv2.putText(image,str(str(sim_comp[2][1])+' '+(str(sim_comp[2][0])[::-1][3:][::-1])), (int(wristMarks.x*ix)+20,int(wristMarks.y*iy)+60), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),2)
                    replace(canvas,spriteBase[sim_comp[0][1]].out(),(int(wristMarks.x*bgx)-xtest.get(),int(wristMarks.y*bgy)+ytest.get()))
                    #if depth:
                        #cv2.circle(canvas,(int(wristMarks.x*bgx)-xtest.get(),int(wristMarks.y*bgy)+ytest.get()), max(1,int(abs(wristMarks.z)*spriteSizeFac.get())), (0,0,255), -1)
                    #else:
                        #cv2.circle(canvas,(int(wristMarks.x*bgx)-xtest.get(),int(wristMarks.y*bgy)+ytest.get()), spriteSizeFac.get(), (0,0,255), -1)
            # Flip the image horizontally for a selfie-view display.
            #cv2.imshow('MediaPipe Pose', image)
            cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow("window", canvas)
            #cv2.imshow('Display', cv2.resize(cv2.flip(canvas,1),dsize=(192*3,108*3)))
            if cv2.waitKey(5) & 0xFF == 27:
                setting.destroy()
                cap.release()
                break
mainThread = Thread(target = cvMainLoop)
mainThread.setDaemon(True)
mainThread.start()
setting.mainloop()
cap.release()