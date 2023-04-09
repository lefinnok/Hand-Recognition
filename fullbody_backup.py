import cv2
import mediapipe as mp
from cam_ports import list_ports
import numpy as np
import networkx as nx
import glob
import re
mp_drawing = mp.solutions.drawing_utils
drawer = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.1,min_tracking_confidence=0.1)
# For webcam input:
cap = cv2.VideoCapture(2)
#Setting Values
spriteSizeFac = 100
depth = False

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
      15:{11,0},
      19:{11,0},
      16:{12,0},
      16:{20,0},
      14:{24,0},
      13:{23,0},
      28:{27,0},
      27:{28,0}


    }
    
    
    
    adjecentcy_list = []
    
    reference_vector = np.array([landmarks[1].x,landmarks[1].y,landmarks[1].z])-np.array([landmarks[0].x,landmarks[0].y,landmarks[0].z])
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
    expr = re.compile("(?<=poses\/)[^.]*")
    name = expr.search(img).group(0)
    ref_images[name] = n
    print(name)
#exit()

ref_pGs = {}

for ref_img in ref_images:
    ref_height, ref_width, ref_channels = ref_images[ref_img].shape 
    rgb_ref_img = cv2.cvtColor(ref_images[ref_img],cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_ref_img)
    
    
    #print("image")
    if results.pose_landmarks:
        
        #drawer.draw_landmarks(ref_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        #cv2.imshow('stuff',ref_img)
        #cv2.waitKey(0)
        ref_pGs[ref_img] = bod2Graph(results.pose_landmarks)


with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    canvas = np.zeros(image.shape, np.uint8);
    ix,iy,iz = image.shape
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
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        wristMarks = landmarks[0]
        if wristMarks:
          cur_pG = bod2Graph(results.pose_landmarks)
          sim_comp = [(int(graphSim(ref_pGs[ref_pG],cur_pG)),ref_pG) for ref_pG in ref_pGs]
          
          cv2.putText(image,str(min(sim_comp,key=lambda x:x[0])), (int(wristMarks.x*ix)+20,int(wristMarks.y*iy)), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),2)
          if depth:
            cv2.circle(canvas,(int(wristMarks.x*ix),int(wristMarks.y*iy)), max(1,int(abs(wristMarks.z)*spriteSizeFac)), (0,0,255), -1)
          else:
            cv2.circle(canvas,(int(wristMarks.x*ix),int(wristMarks.y*iy)), spriteSizeFac, (0,0,255), -1)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', image)
    #cv2.imshow('Display', cv2.flip(canvas,1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()