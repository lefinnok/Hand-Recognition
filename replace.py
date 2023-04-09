import cv2
import glob
import numpy as np
#a = cv2.imread('sprites/bg/Background_Blurrier-1.png',cv2.IMREAD_UNCHANGED)
#b = cv2.imread('sprites/flower/Foreground_Flower-74.png',cv2.IMREAD_UNCHANGED)
def replace(imga,imgb,coords,center=True):
    cx,cy = coords
    iay,iax,_ = imga.shape
    iby,ibx,_ = imgb.shape
    if center:
        cx-=(ibx>>1)
        cy-=(iby>>1)
    for row in range(iby):
        if (row+cy)>=iay or (row+cy)<0:
            continue
        for column in range(ibx):
            if (column+cx)>=iax or (column+cx)<0:
                continue
            if imgb[row][column][3]:
                imga[cy+row][cx+column] = imgb[row][column]
                

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

#replace(a,cv2.resize(b,dsize=(200,200)),(800,800),True)
#replace(a,cv2.resize(b,dsize=(200,200)),(800,800),False)
#cv2.imshow('x',cv2.resize(cv2.add(a,cv2.resize(b,dsize=(1920,1080))),dsize=(192,108)))
#cv2.imshow('x',cv2.resize(a,dsize=(192*2,108*2)))
#cv2.waitKey(0)
