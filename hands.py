# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 14:50:34 2021

@author: lefin
"""

import cv2
import mediapipe as mp
import time
import numpy as np
import networkx as nx
import glob
from math import acos

'''
notes:
    I've noticed from experimental trials, that connections/structure of the
    hand graph in the hand2graph function heavily influenced the accuracy of
    the detection of similatiries between two graphs. connections between
    heavily moving nodes (non piviting like finger tips) are key to the process
    key 
    
    another factor is the minimum_energy for k, the larger the k, the more 
    accurate the solution is.'
'''

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




hand_det = mp.solutions.hands

hands = hand_det.Hands(False, 2)

drawer = mp.solutions.drawing_utils


        

def hand2Graph(hand_result, scale = 100):
    landmarks = hand_result.landmark
    G = nx.Graph()
    G.add_nodes_from(list(range(21)))
    
    
    
    
    '''
    #tested connection structures (ranked from least to most accurate)
    
    #unaltered connections
    struct_dict = {0:{1,5,9},
                   1:{2},
                   2:{3},
                   3:{4},
                   5:{6,9},
                   6:{7},
                   7:{8},
                   9:{10,13},
                   10:{11},
                   11:{12},
                   13:{14,17},
                   14:{15},
                   15:{16},
                   17:{18},
                   18:{19},
                   19:{20}}
    
    
    #full connection to root
    struct_dict = {0:{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},
                   1:{2},
                   2:{3},
                   3:{4},
                   5:{6,9},
                   6:{7},
                   7:{8},
                   9:{10,13},
                   10:{11},
                   11:{12},
                   13:{14,17},
                   14:{15},
                   15:{16},
                   17:{18},
                   18:{19},
                   19:{20}}
    
    #only root to base and base to tips and between tips (nope, bases are not good for accuracy)
    struct_dict = {0:{1,5,9,13,17},
                   
                   4:{1,8},
                   
                   8:{5,12},    
                   
                   12:{9,16},
                   
                   16:{13,20},
                   
                   20:{17}
                   }
    
    
    
    #full connection to root and full between tips
    struct_dict = {0:{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},
                   1:{2},
                   2:{3},
                   3:{4},
                   4:{8,12,16,20},
                   5:{6,9},
                   6:{7},
                   7:{8},
                   8:{12,16,20},
                   9:{10,13},
                   10:{11},
                   11:{12},
                   12:{16,20},
                   13:{14,17},
                   14:{15},
                   15:{16},
                   16:{20},
                   17:{18},
                   18:{19},
                   19:{20}}
    
    
    '''
    #full connection to root and between tips
    struct_dict = {0:{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},
                   1:{2},
                   2:{3},
                   3:{4},
                   4:{8},
                   5:{6,9},
                   6:{7},
                   7:{8},
                   8:{12},
                   9:{10,13},
                   10:{11},
                   11:{12},
                   12:{16},
                   13:{14,17},
                   14:{15},
                   15:{16},
                   16:{20},
                   17:{18},
                   18:{19},
                   19:{20}}
    
    
    
    #only full connection to root and between tips
    struct_dict = {0:{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},
                   
                   4:{8},
                   
                   8:{12},
                   
                   12:{16},
                   
                   16:{20},
                   }
    
    #only root and full between tips 
    struct_dict = {4:{0,8,12,16,20},
                   
                   8:{0,12,16,20},    
                   
                   12:{0,16,20},
                   
                   16:{0,20},
                   
                   20:{0}
                   }
    
    #only root and between tips (much more accurate but less lee-way)
    struct_dict = {4:{0,8},
                   
                   8:{0,12},    
                   
                   12:{0,16},
                   
                   16:{0,20},
                   
                   20:{0}
                   }
    
    #only root and between tips + thumb and pinky
    struct_dict = {4:{0,8,20},
                   
                   8:{0,12},    
                   
                   12:{0,16},
                   
                   16:{0,20},
                   
                   20:{0}
                   }
    
    #only root and between tips + thumb and all (accurate for all except between 7 and 6)
    struct_dict = {4:{0,8,12,16,20},
                   
                   8:{0,12},    
                   
                   12:{0,16},
                   
                   16:{0,20},
                   
                   20:{0}
                   }
    
    
    
    
    adjecentcy_list = []
    
    reference_vector = np.array([landmarks[5].x,landmarks[5].y,landmarks[5].z])-np.array([landmarks[0].x,landmarks[0].y,landmarks[0].z])
    size_factor = np.linalg.norm(reference_vector)
    
    
    for origin in struct_dict:
        or_landmark = landmarks[origin]
        or_ary = np.array([or_landmark.x,or_landmark.y,or_landmark.z])*scale/size_factor
        for target in struct_dict[origin]:
            tar_landmark = landmarks[target]
            tar_ary = np.array([tar_landmark.x,tar_landmark.y,tar_landmark.z])*scale/size_factor
            #weight is the magnitude of the vector(obtained by subtracting the two positional vectors)
            
            #incoporating the angle into the calculation is not ideal
            a,b = reference_vector,tar_ary-or_ary
            dot_product = np.dot(a,b)
            
            angle = acos(dot_product/(np.linalg.norm(a)*np.linalg.norm(b)))
            
            if type(angle) != float:
                print(angle)
                angle = 1
            
            angle = 1
            ''''''
            
            adjecentcy_list.append((origin,target,np.linalg.norm(tar_ary-or_ary)*angle))
    '''
    for origin in range(21):
       or_landmark = landmarks[origin]
       or_ary = np.array([or_landmark.x,or_landmark.y,or_landmark.z])*scale/size_factor
       n = list(range(21))
       n.remove(origin)
       for target in n:
           tar_landmark = landmarks[target]
           tar_ary = np.array([tar_landmark.x,tar_landmark.y,tar_landmark.z])*scale/size_factor
           #weight is the magnitude of the vector(obtained by subtracting the two positional vectors)
           adjecentcy_list.append((origin,target,np.linalg.norm(tar_ary-or_ary)))
   '''
    G.add_weighted_edges_from(adjecentcy_list)
    
    return G




# load and process reference images

filenames = [img for img in glob.glob("images/*.jpg")]

filenames.sort() # ADD THIS LINE

ref_images = []
for img in filenames:
    n= cv2.imread(img)
    ref_images.append(n)


ref_hGs = []

for ref_img in ref_images:
    ref_height, ref_width, ref_channels = ref_img.shape 
    rgb_ref_img = cv2.cvtColor(ref_img,cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_ref_img)
    
    
    
    if results.multi_hand_landmarks:
        for hand_result in results.multi_hand_landmarks:
            drawer.draw_landmarks(ref_img, hand_result, hand_det.HAND_CONNECTIONS)
            ref_hGs.append(hand2Graph(hand_result))
    
    
    






ptime, ctime = 0, 0

'''

'''

cap = cv2.VideoCapture(0)

results = None

while True:
    #fps calculation
    ctime = time.time()
    fps = int(1/(ctime - ptime))
    ptime = ctime
    
    #image processing
    run, img = cap.read()
    height, width, channels = img.shape 
    rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)
    
    if results.multi_hand_landmarks:
        for hand_result in results.multi_hand_landmarks:
            drawer.draw_landmarks(img, hand_result, hand_det.HAND_CONNECTIONS)
            
            landmarks = hand_result.landmark
            wrist,mid_base = (int(landmarks[0].x*width),int(landmarks[0].y*height)),(int(landmarks[9].x*width),int(landmarks[9].y*height))
            palm_location = np.add(np.array(mid_base),np.subtract(np.array(wrist), np.array(mid_base))/2).astype(int)
            cv2.circle(img,  tuple(palm_location), 5, (255,0,255), -1)
            
            cur_hG = hand2Graph(hand_result)
            
            sim_comp = [(idx,int(graphSim(ref_hG,cur_hG))) for idx, ref_hG in enumerate(ref_hGs,1)]
            
            #min(sim_comp,key=lambda x:x[1])[0]
            print(sim_comp)
            
            cv2.putText(img,str(min(sim_comp,key=lambda x:x[1])), tuple(palm_location), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255),1)
            
    
    
    cv2.putText(img,str(fps), (50,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255),2)
    
    
    
    #show image
    if run:
        cv2.imshow('Hand', img)
    
    if cv2.waitKey(10) == ord('q'):
        print('exitting loop')
        cap.release()
        cv2.destroyAllWindows()
        if results.multi_hand_landmarks:
            results = results.multi_hand_landmarks 
        break

for result in results:
    print(result)
    print(result.landmark)
''''''

''''''