import cv2
import numpy as np

#location img
pics_path='./pics/'
name = 'equus'
mypic=pics_path+name+'.jpg'

#Var Pyr - bilateral
num_iter = 1      #iteration pirUp/donw
num_bilateral = 8  # number of bilateral filtering steps


#TRAITEMENT IMAGE COULEUR
img_rgb = cv2.imread(mypic)

#flou lissage couleurs
img_color = img_rgb
for i in xrange(num_iter):
    img_color = cv2.pyrDown(img_color)
 
 
#BilateralFilter en 8 iterations 
for i in xrange(num_bilateral):
    img_color = cv2.bilateralFilter(img_color, d=9,
                                    sigmaColor=7,
                                    sigmaSpace=7)
# pyrUp retour la taille d'origine
for i in xrange(num_iter):
    img_color = cv2.pyrUp(img_color)

#TRAITEMENT IMAGE N&B

img_rgb = cv2.imread(mypic)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

#inversion couleurs
img_gray_inv = 255 - img_gray

img_fl = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21),
      	                     sigmaX=0, sigmaY=0)

#division - facteur scalaire 256
img_bl = cv2.divide(img_gray, 255-img_fl, scale=256)

# recuperation et amelioration des contour
img_contour = cv2.adaptiveThreshold(img_bl, 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY,
                                 blockSize=21,
                                 C=10)

#img_contour = cv2.dilate(img_contour, np.ones((255,12)))
#conversion en couleur pour fusionner les deux images
img_contour = cv2.cvtColor(img_contour, cv2.COLOR_GRAY2RGB)
img_dessin = cv2.bitwise_and(img_color, img_contour)

# save - dossier pics
cv2.imwrite(pics_path+name+'_drwing1.jpg', img_dessin)
