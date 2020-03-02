#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

# Cria o detector BRISK
brisk = cv2.BRISK_create()

# Configura o algoritmo de casamento de features que vê *como* o objeto que deve ser encontrado aparece na imagem
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Define o mínimo de pontos similares
MINIMO_SEMELHANCAS = 18
    
def find_homography_draw_box(kp1, kp2, img_cena):
    
    out = img_cena.copy()
    
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    # Tenta achar uma trasformacao composta de rotacao, translacao e escala que situe uma imagem na outra
    # Esta transformação é chamada de homografia 
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img_original.shape
    
    # Um retângulo com as dimensões da imagem original
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    # Transforma os pontos do retângulo para onde estao na imagem destino usando a homografia encontrada
    dst = cv2.perspectiveTransform(pts,M)

    # Desenha um contorno em vermelho ao redor de onde o objeto foi encontrado
    img2b = cv2.polylines(out,[np.int32(dst)],True,(255,255,0),5, cv2.LINE_AA)
    
    return img2b

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    original_rgb = cv2.imread("nome_insper.png")  # Imagem a procurar
    img_original = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2GRAY)

    # Encontra os pontos únicos (keypoints) nas duas imagems
    kp1, des1 = brisk.detectAndCompute(img_original ,None)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret == False:
            print("Problema para capturar o frame da câmera")
            continue
        
        framed = None

        # Our operations on the frame come here
        frame_rgb = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        kp2, des2 = brisk.detectAndCompute(gray,None)

        # Tenta fazer a melhor comparacao usando o algoritmo 
        matches = bf.knnMatch(des1,des2,k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good) > MINIMO_SEMELHANCAS:
            framed = find_homography_draw_box(kp1, kp2, frame)
            cv2.imshow('BRISK features', framed)
        else:
            cv2.imshow("BRISK features", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

