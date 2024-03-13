import cv2 as cv
import os
import numpy as np
import math
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
import pandas as pd

def calcEnergy(glcm):
	return graycoprops(glcm, 'energy')[0, 0]

def calcContrast(glcm):
	return graycoprops(glcm, 'contrast')[0, 0]

def calcHomogeneity(glcm):
	return graycoprops(glcm, 'homogeneity')[0, 0]

def calcCorrelation(glcm):
	return graycoprops(glcm, 'correlation')[0, 0]

def calcGLCM(gray, theta = 0, dist = 5):
	return graycomatrix(gray, distances=[dist], angles=[theta], levels= 256, symmetric=True, normed=True)

def openImagesFromFolder(folder):

	images = []
	liste = os.listdir(folder)
	for i in range( 0, len(liste) ):
		if liste[i].endswith(".jpg") or liste[i].endswith(".jpeg") or liste[i].endswith(".png") or liste[i].endswith(".bmp"):
			images.append(cv.imread( folder + "/" + liste[i] ) )

	return liste,images

def calcDescripteurTexture(imgs, dist = 5, theta = 0):
	descripteurs = []
	for i in range (0, len(imgs)):
		gray = cv.cvtColor(imgs[i], cv.COLOR_BGR2GRAY)
		glcm = calcGLCM(gray, theta = theta, dist = dist)

		en = calcEnergy(glcm)
		ct = calcContrast(glcm) / (imgs[i].shape[0] * imgs[i].shape[1])
		ho = calcHomogeneity(glcm)
		cr = (calcCorrelation(glcm) + 1) / 2.0

		descripteurs.append([en, ct, cr, ho])

	return np.array(descripteurs)

def calcDescripteurColorRGB(imgs):
	descripteurs = []
	white = np.matrix([255,255,255],np.uint8)
	for i in range (0, len(imgs)):
		bgr = np.matrix([255,255,255], np.uint8)
		for x in range(imgs[i].shape[0]):
			for y in range(imgs[i].shape[1]):
				if ( not np.array_equiv(imgs[i][x,y,:], white) ):
					bgr = imgs[i][x,y,:]

		descripteurs.append( [ bgr[2], bgr[1], bgr[0] ] )

	return np.array(descripteurs)/255.0

def calcDescripteurColorYCbCr(imgs):
	descripteurs = []
	white = np.matrix([255,128,128],np.uint8)
	for i in range (0, len(imgs)):
		imgs[i] = cv.cvtColor(imgs[i], cv.COLOR_BGR2YCrCb)
		ycbcr = np.matrix([255,255,255], np.uint8)
		for x in range(imgs[i].shape[0]):
			for y in range(imgs[i].shape[1]):
				if ( not np.array_equiv(imgs[i][x,y,:], white) ):
					ycbcr = imgs[i][x,y,:]

		descripteurs.append( [ ycbcr[0], ycbcr[1], ycbcr[2] ] )

	return np.array(descripteurs)/255.0

def knn( categories, test, k = 1 ):

	tab = np.full(k, np.inf)
	ks = np.full(k, 0)

	for cl in range( 0, len(categories) ):
		for des in categories[cl]:
			d = 0
			for i in range( 0, len(des) ):
				d += (des[i] - test[i])**2

			if ( d <= max(tab) ):
				tab[ np.where( tab == max(tab) )[0][0] ] = d
				ks[ np.where( tab == max(tab) )[0][0] ] = cl

	res = np.argmax(np.bincount(ks))
	return res

def nuee_dynamique(desc, k=2):
    if k <= 1:
        print("k must be strictly over 1")
        return -1

    markers = []
    cla = np.zeros(len(desc), dtype=np.uint8)

    for i in range(k):
        markers.append(desc[i])

    markers = np.array(markers)

    def rec_nuee(desc, markers, cla):
        new = np.zeros(len(desc), dtype=np.uint8)

        for i in range(len(desc)):
            sd = None
            for j in range(len(markers)):
                d = 0
                for k in range(len(desc[i])):
                    d += (desc[i, k] - markers[j, k]) ** 2

                if sd is None or sd > d:
                    sd = d
                    new[i] = j

        if np.array_equal(new, cla):
            return markers, new
        else:
            new_markers = np.zeros(markers.shape, dtype=float)

            for i in range(len( desc[0]) ):
                for j in range( len(desc) ):
                	for k in range( len(markers) ):
	                    if new[j] == k:
	                        new_markers[k, i] += desc[j, i]

            for k in range( len(markers) ):
                val = np.count_nonzero(new == k)
                if val != 0:
                    new_markers[k, :] /= val
                else:
                    new_markers[k, :] = markers[k, :]

            return rec_nuee(desc, new_markers, new)

    return rec_nuee(desc, markers, cla)


###################[DESCRIPTEUR TEXTURE]##################
nameClass1, class1 = openImagesFromFolder("spec/classe1")
nameClass2, class2 = openImagesFromFolder("spec/classe2")
nameTest, test = openImagesFromFolder("spec/test")
names, textures = openImagesFromFolder("spec/imgTexture")

#Plots based on energy and contrast
columns = 3
rows = 1

for d in range(0,2):
	fig = plt.figure(figsize = (12,4))
	for t in range (0,3):
		o = t*np.pi/4
		desC1 = calcDescripteurTexture(class1, dist = 4**d, theta = o)
		desC2 = calcDescripteurTexture(class2, dist = 4**d, theta = o)
		desT = calcDescripteurTexture(test, dist = 4**d, theta = o)
		desAll = calcDescripteurTexture(textures, dist = 4**d, theta = o)

		df = pd.DataFrame(
			{
				'Name' 			: 	names,
				'energy' 		:	desAll[:,0],
				'contrast' 		:	desAll[:,1],
				'correlation' 	:	desAll[:,2],
				'homogeneity' 	:	desAll[:,3],
			}
		)

		df.to_csv(f"csv/dist_{4**d}/theta_{t*45}deg.csv", index = False)

		desR1 = []
		desR2 = []

		for i in range(len(desT)):
			if ( knn([desC1[:,:2], desC2[:,:2]], desT[i,:2], k = 3) == 0 ):
				desR1.append(desT[i])
			else:
				desR2.append(desT[i])

		desR1 = np.array(desR1)
		desR2 = np.array(desR2)

		ax = fig.add_subplot(rows, columns, 1 + t)
		ax.set_xlabel('energy')
		ax.set_ylabel('contrast')

		ax.autoscale(False)
		mx = max( np.max(desC1[:,:2]), np.max(desC2[:,:2]), np.max(desT[:,:2]) ) * 1.1
		mi = min( np.min(desC1[:,:2]), np.min(desC2[:,:2]), np.min(desT[:,:2]) )
		mi -= abs(mx * 0.1)
		ax.axis([mi, mx, mi, mx])

		ax.scatter(desC1[:,0], desC1[:,1], c = 'darkred', marker='o')
		ax.scatter(desC2[:,0], desC2[:,1], c = 'dodgerblue', marker='o')
		if ( len(desR1) > 0 ):
			ax.scatter(desR1[:,0], desR1[:,1], c = 'red', marker='^')
		if ( len(desR2) > 0 ):
			ax.scatter(desR2[:,0], desR2[:,1], c = 'blue', marker='^')

		plt.title(f"d = {4**d} and theta = {t*45}deg")
		plt.gca().set_facecolor((0.8, 0.8, 0.8))

	plt.subplots_adjust(bottom=0.15, left=0.10, right=0.90, top=0.90, wspace = 0.30, hspace = 0.25)
	plt.show()

#Nuee Dynamique (2D descriptor (Energy, Contrast))
names, imgs = openImagesFromFolder("spec/imgTexture")
dts = calcDescripteurTexture(imgs)

fig = plt.figure(figsize = (13,7))
columns = 2
rows = 2
for nk in range (2, 6):

	markers, klass = nuee_dynamique( dts[:,:2], k = nk )
	
	ax = fig.add_subplot(rows, columns, nk-1)
	ax.set_xlabel('energy')
	ax.set_ylabel('contrast')
	ax.autoscale(False)
	mx = np.max(dts[:,:2]) * 1.1
	mi = np.min(dts[:,:2])
	mi -= abs(mx * 0.1)
	ax.axis([0, mx, 0, mx])
	for i in range( 0, len(dts) ):
		if klass[i] == 0:
			ax.scatter(dts[i,0], dts[i,1], c = 'darkred', marker='o')
		if klass[i] == 1:
			ax.scatter(dts[i,0], dts[i,1], c = 'dodgerblue', marker='o')
		if klass[i] == 2:
			ax.scatter(dts[i,0], dts[i,1], c = 'green', marker='o')
		if klass[i] == 3:
			ax.scatter(dts[i,0], dts[i,1], c = 'goldenrod', marker='o')
		if klass[i] == 4:
			ax.scatter(dts[i,0], dts[i,1], c = 'darkviolet', marker='o')

	#Plotting the markers
	ax.scatter(markers[0,0], markers[0,1], c = 'red', marker='^')
	if (nk > 1):
		ax.scatter(markers[1,0], markers[1,1], c = 'blue', marker='^')
	if (nk > 2):
		ax.scatter(markers[2,0], markers[2,1], c = 'lime', marker='^')
	if (nk > 3):
		ax.scatter(markers[3,0], markers[3,1], c = 'yellow', marker='^')
	if (nk > 4):
		ax.scatter(markers[4,0], markers[4,1], c = 'violet', marker='^')

	plt.title(f"Energy and Contrast Based Dynamic Clustering (n={nk})")
	plt.gca().set_facecolor((0.2, 0.2, 0.2))

plt.subplots_adjust(bottom=0.15, left=0.15, right=0.90, top=0.90, wspace = 0.25, hspace = 0.25)
plt.show()


#Nuee Dynamique (3D descriptor (Energy, Contrast, correlation))
names, imgs = openImagesFromFolder("spec/imgTexture")
dts = calcDescripteurTexture(imgs)

markers, klass = nuee_dynamique( dts[:,:3], k = 5 )

fig = plt.figure(figsize = (7,7))
columns = 1
rows = 1
ax = fig.add_subplot(1, 1, 1, projection = '3d')
ax.set_xlabel('energy')
ax.set_ylabel('contrast')
ax.set_zlabel('correlation')
ax.autoscale(False)
mx = np.max(dts[:,:3]) * 1.1
mi = np.min(dts[:,:3])
mi -= abs(mx * 0.1)
ax.set_xlim([mi, mx])
ax.set_ylim([mi, mx])
ax.set_zlim([mi, mx])
for i in range( 0, len(dts) ):
	if klass[i] == 0:
		ax.scatter(dts[i,0], dts[i,1], dts[i,2], c = 'darkred', marker='o')
	if klass[i] == 1:
		ax.scatter(dts[i,0], dts[i,1], dts[i,2], c = 'dodgerblue', marker='o')
	if klass[i] == 2:
		ax.scatter(dts[i,0], dts[i,1], dts[i,2], c = 'green', marker='o')
	if klass[i] == 3:
		ax.scatter(dts[i,0], dts[i,1], dts[i,2], c = 'goldenrod', marker='o')
	if klass[i] == 4:
		ax.scatter(dts[i,0], dts[i,1], dts[i,2], c = 'darkviolet', marker='o')

#Plotting the markers
ax.scatter(markers[0,0], markers[0,1], markers[0,2], c = 'red', marker='^')
ax.scatter(markers[1,0], markers[1,1], markers[1,2], c = 'blue', marker='^')
ax.scatter(markers[2,0], markers[2,1], markers[2,2], c = 'lime', marker='^')
ax.scatter(markers[3,0], markers[3,1], markers[3,2], c = 'yellow', marker='^')
ax.scatter(markers[4,0], markers[4,1], markers[4,2], c = 'violet', marker='^')

plt.title(f"Energy and Contrast and Correlation Based Dynamic Clustering")
plt.gca().set_facecolor((0.8, 0.8, 0.8))

plt.subplots_adjust(bottom=0.15, left=0.15, right=0.90, top=0.90, wspace = 0.25, hspace = 0.25)
plt.show()


###################[DESCRIPTEUR COULEUR]##################
nameClass1, class1 = openImagesFromFolder("spec/color1")
nameClass2, class2 = openImagesFromFolder("spec/color2")
nameTest, test = openImagesFromFolder("spec/testCouleur")
names, colors = openImagesFromFolder("spec/imgCouleur")

#Plots based on Cb and Cr
desC1 = calcDescripteurColorYCbCr(class1)
desC2 = calcDescripteurColorYCbCr(class2)
desT = calcDescripteurColorYCbCr(test)
desYCB = calcDescripteurColorYCbCr(colors)

df = pd.DataFrame(
	{
		'Name' 	: 	names,
		'Y' 	:	desYCB[:,0],
		'Cb' 	:	desYCB[:,1],
		'Cr' 	:	desYCB[:,2],
	}
)

df.to_csv(f"csv/color/YCbCr.csv", index = False)

desR1 = []
desR2 = []

for i in range(len(desT)):
	if ( knn([desC1[:,1:], desC2[:,1:]], desT[i,1:], k = 3) == 0 ):
		desR1.append(desT[i])
	else:
		desR2.append(desT[i])

desR1 = np.array(desR1)
desR2 = np.array(desR2)

fig = plt.figure(figsize = (7, 7))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Cb')
ax.set_ylabel('Cr')

ax.autoscale(False)
mx = 1.2#max( np.max(desC1[:,1:]), np.max(desC2[:,1:]), np.max(desT[:,1:]) )
mi = -0.2
ax.axis([mi, mx, mi, mx])

ax.scatter(desC1[:,0], desC1[:,1], c = 'darkred', marker='o')
ax.scatter(desC2[:,0], desC2[:,1], c = 'dodgerblue', marker='o')
if ( len(desR1) > 0 ):
	ax.scatter(desR1[:,0], desR1[:,1], c = 'red', marker='^')
if ( len(desR2) > 0 ):
	ax.scatter(desR2[:,0], desR2[:,1], c = 'blue', marker='^')

plt.title(f"KNN Couleur")
plt.gca().set_facecolor((0.8, 0.8, 0.8))

plt.subplots_adjust(bottom=0.15, left=0.10, right=0.90, top=0.90, wspace = 0.30, hspace = 0.25)
plt.show()

#Nuee Dynamique (2D descriptor (Cb, Cr))
names, imgs = openImagesFromFolder("spec/imgCouleur")
dts = calcDescripteurColorYCbCr(imgs)

fig = plt.figure(figsize = (13,7))
columns = 2
rows = 2
for nk in range (2, 6):

	markers, klass = nuee_dynamique( dts[:,1:], k = nk )
	ax = fig.add_subplot(rows, columns, nk-1)
	ax.set_xlabel('Cb')
	ax.set_ylabel('Cr')
	ax.autoscale(False)
	mx = np.max(dts[:,1:]) * 1.1
	mi = np.min(dts[:,1:])
	mi -= abs(mx * 0.1)
	ax.axis([0, mx, 0, mx])
	for i in range( 0, len(dts) ):
		if klass[i] == 0:
			ax.scatter(dts[i,1], dts[i,2], c = 'darkred', marker='o')
		if klass[i] == 1:
			ax.scatter(dts[i,1], dts[i,2], c = 'dodgerblue', marker='o')
		if klass[i] == 2:
			ax.scatter(dts[i,1], dts[i,2], c = 'green', marker='o')
		if klass[i] == 3:
			ax.scatter(dts[i,1], dts[i,2], c = 'goldenrod', marker='o')
		if klass[i] == 4:
			ax.scatter(dts[i,1], dts[i,2], c = 'darkviolet', marker='o')

	#Plotting the markers
	ax.scatter(markers[0,0], markers[0,1], c = 'red', marker='^')
	if (nk > 1):
		ax.scatter(markers[1,0], markers[1,1], c = 'blue', marker='^')
	if (nk > 2):
		ax.scatter(markers[2,0], markers[2,1], c = 'lime', marker='^')
	if (nk > 3):
		ax.scatter(markers[3,0], markers[3,1], c = 'yellow', marker='^')
	if (nk > 4):
		ax.scatter(markers[4,0], markers[4,1], c = 'violet', marker='^')

	plt.title(f"Cb, Cr Based Dynamic Clustering (n={nk})")
	plt.gca().set_facecolor((0.2, 0.2, 0.2))

plt.subplots_adjust(bottom=0.15, left=0.15, right=0.90, top=0.90, wspace = 0.275, hspace = 0.275)
plt.show()


#Nuee Dynamique (3D descriptor (R,G,B))
names, imgs = openImagesFromFolder("spec/imgCouleur")
dts = calcDescripteurColorRGB(imgs)

df = pd.DataFrame(
	{
		'Name' 	: 	names,
		'R' 	:	dts[:,0],
		'G' 	:	dts[:,1],
		'B' 	:	dts[:,2],
	}
)

df.to_csv(f"csv/color/RGB.csv", index = False)

markers, klass = nuee_dynamique( dts[:,:3], k = 5 )

fig = plt.figure(figsize = (7,7))
columns = 1
rows = 1
ax = fig.add_subplot(1, 1, 1, projection = '3d')
ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')
ax.autoscale(False)
mx = np.max(dts[:,:3])
mi = np.min(dts[:,:3])
ax.set_xlim([mi, mx])
ax.set_ylim([mi, mx])
ax.set_zlim([mi, mx])
for i in range( 0, len(dts) ):
	if klass[i] == 0:
		ax.scatter(dts[i,0], dts[i,1], dts[i,2], c = 'darkred', marker='o')
	if klass[i] == 1:
		ax.scatter(dts[i,0], dts[i,1], dts[i,2], c = 'dodgerblue', marker='o')
	if klass[i] == 2:
		ax.scatter(dts[i,0], dts[i,1], dts[i,2], c = 'green', marker='o')
	if klass[i] == 3:
		ax.scatter(dts[i,0], dts[i,1], dts[i,2], c = 'goldenrod', marker='o')
	if klass[i] == 4:
		ax.scatter(dts[i,0], dts[i,1], dts[i,2], c = 'darkviolet', marker='o')

#Plotting the markers
ax.scatter(markers[0,0], markers[0,1], markers[0,2], c = 'red', marker='^')
ax.scatter(markers[1,0], markers[1,1], markers[1,2], c = 'blue', marker='^')
ax.scatter(markers[2,0], markers[2,1], markers[2,2], c = 'lime', marker='^')
ax.scatter(markers[3,0], markers[3,1], markers[3,2], c = 'yellow', marker='^')
ax.scatter(markers[4,0], markers[4,1], markers[4,2], c = 'violet', marker='^')

plt.title(f"RGB Based Dynamic Clustering")
plt.gca().set_facecolor((0.8, 0.8, 0.8))

plt.subplots_adjust(bottom=0.15, left=0.15, right=0.90, top=0.90, wspace = 0.25, hspace = 0.25)
plt.show()

