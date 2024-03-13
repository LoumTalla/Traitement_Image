import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

flat3 = np.matrix([[1, 1, 1],
				   [1, 1, 1],
				   [1, 1, 1]], np.int8)

boule3 = np.matrix([[0, 1, 0],
				    [1, 1, 1],
				    [0, 1, 0]], np.int8)

def erosion(img, mat):

	out = np.zeros(img.shape)

	for i in range( 0, img.shape[0] ):
		for j in range ( 0, img.shape[1] ):

			v = None
			k0 = math.floor(mat.shape[0]/2)
			k1 = math.floor(mat.shape[1]/2)

			if ( i >= k0 and i < img.shape[0] - k0 ):

				if ( j >= k1 and j < img.shape[1] - k1 ):

					for x in range (0, mat.shape[0]):
						for y in range (0, mat.shape[1]):

							if (mat[x,y] == 1):
								if (v == None or v > img[ i + x - 1, j + y - 1 ]):
									v = img[ i + x - 1, j + y - 1 ]

				elif ( j < k1):

					for x in range (0, mat.shape[0]):
						for y in range (k1 - j, mat.shape[1]):

							if (mat[x,y] == 1):
								if (v == None or v > img[ i + x - 1, j + y - 1 ]):
									v = img[ i + x - 1, j + y - 1 ]

				elif (j >= img.shape[1] - k1):

					for x in range (0, mat.shape[0]):
						for y in range (0, img.shape[1] - j + k1):

							if (mat[x,y] == 1):
								if (v == None or v > img[ i + x - 1, j + y - 1 ]):
									v = img[ i + x - 1, j + y - 1 ]

			###################################################################
			elif ( i < k0 ):

				if ( j >= k1 and j < img.shape[1] - k1):

					for x in range ( k0 - i, mat.shape[0]):
						for y in range (0, mat.shape[1]):

							if (mat[x,y] == 1):
								if (v == None or v > img[ i + x - 1, j + y - 1 ]):
									v = img[ i + x - 1, j + y - 1 ]

				elif ( j < k1):

					for x in range ( k0 - i, mat.shape[0]):
						for y in range (k1 - j, mat.shape[1]):

							if (mat[x,y] == 1):
								if (v == None or v > img[ i + x - 1, j + y - 1 ]):
									v = img[ i + x - 1, j + y - 1 ]

				elif (j >= img.shape[1] - k1):

					for x in range ( k0 - i, mat.shape[0]):
						for y in range (0, img.shape[1] - j + k1):

							if (mat[x,y] == 1):
								if (v == None or v > img[ i + x - 1, j + y - 1 ]):
									v = img[ i + x - 1, j + y - 1 ]

			####################################################################
			elif ( i >= img.shape[0] - k0):

				if ( j >= k1 and j < img.shape[1] - k1):

					for x in range (0, img.shape[0] - i + k0):
						for y in range (0, mat.shape[1]):

							if (mat[x,y] == 1):
								if (v == None or v > img[ i + x - 1, j + y - 1 ]):
									v = img[ i + x - 1, j + y - 1 ]

				elif ( j < k1):

					for x in range (0, img.shape[0] - i + k0):
						for y in range (k1 - j, mat.shape[1]):

							if (mat[x,y] == 1):
								if (v == None or v > img[ i + x - 1, j + y - 1 ]):
									v = img[ i + x - 1, j + y - 1 ]

				elif (j >= img.shape[1] - k1):

					for x in range (0, img.shape[0] - i + k0):
						for y in range (0, img.shape[1] - j + k1):

							if (mat[x,y] == 1):
								if (v == None or v > img[ i + x - 1, j + y - 1 ]):
									v = img[ i + x - 1, j + y - 1 ]

			out[i,j] = v

	return out

def dilatation(img, mat):

	out = np.zeros(img.shape)

	for i in range( 0, img.shape[0] ):
		for j in range ( 0, img.shape[1] ):

			v = None
			k0 = math.floor(mat.shape[0]/2)
			k1 = math.floor(mat.shape[1]/2)

			if ( i >= k0 and i < img.shape[0] - k0 ):

				if ( j >= k1 and j < img.shape[1] - k1 ):

					for x in range (0, mat.shape[0]):
						for y in range (0, mat.shape[1]):

							if (mat[x,y] == 1):
								if (v == None or v < img[ i + x - 1, j + y - 1 ]):
									v = img[ i + x - 1, j + y - 1 ]

				elif ( j < k1):

					for x in range (0, mat.shape[0]):
						for y in range (k1 - j, mat.shape[1]):

							if (mat[x,y] == 1):
								if (v == None or v < img[ i + x - 1, j + y - 1 ]):
									v = img[ i + x - 1, j + y - 1 ]

				elif (j >= img.shape[1] - k1):

					for x in range (0, mat.shape[0]):
						for y in range (0, img.shape[1] - j + k1):

							if (mat[x,y] == 1):
								if (v == None or v < img[ i + x - 1, j + y - 1 ]):
									v = img[ i + x - 1, j + y - 1 ]

			###################################################################
			elif ( i < k0 ):

				if ( j >= k1 and j < img.shape[1] - k1):

					for x in range ( k0 - i, mat.shape[0]):
						for y in range (0, mat.shape[1]):

							if (mat[x,y] == 1):
								if (v == None or v < img[ i + x - 1, j + y - 1 ]):
									v = img[ i + x - 1, j + y - 1 ]

				elif ( j < k1):

					for x in range ( k0 - i, mat.shape[0]):
						for y in range (k1 - j, mat.shape[1]):

							if (mat[x,y] == 1):
								if (v == None or v < img[ i + x - 1, j + y - 1 ]):
									v = img[ i + x - 1, j + y - 1 ]

				elif (j >= img.shape[1] - k1):

					for x in range ( k0 - i, mat.shape[0]):
						for y in range (0, img.shape[1] - j + k1):

							if (mat[x,y] == 1):
								if (v == None or v < img[ i + x - 1, j + y - 1 ]):
									v = img[ i + x - 1, j + y - 1 ]

			####################################################################
			elif ( i >= img.shape[0] - k0):

				if ( j >= k1 and j < img.shape[1] - k1):

					for x in range (0, img.shape[0] - i + k0):
						for y in range (0, mat.shape[1]):

							if (mat[x,y] == 1):
								if (v == None or v < img[ i + x - 1, j + y - 1 ]):
									v = img[ i + x - 1, j + y - 1 ]

				elif ( j < k1):

					for x in range (0, img.shape[0] - i + k0):
						for y in range (k1 - j, mat.shape[1]):

							if (mat[x,y] == 1):
								if (v == None or v < img[ i + x - 1, j + y - 1 ]):
									v = img[ i + x - 1, j + y - 1 ]

				elif (j >= img.shape[1] - k1):

					for x in range (0, img.shape[0] - i + k0):
						for y in range (0, img.shape[1] - j + k1):

							if (mat[x,y] == 1):
								if (v == None or v < img[ i + x - 1, j + y - 1 ]):
									v = img[ i + x - 1, j + y - 1 ]

			out[i,j] = v

	return out

def ouverture(img, mat):
	return dilatation(erosion(img,mat), mat)

def fermeture(img, mat):
	return erosion(dilatation(img,mat), mat)

def lantujoul(img, mat, n):

	i0 = img
	tmp = i0 - ouverture(i0, mat)

	for i in range (0, n):
		i0 = erosion(i0, mat)
		tmp += i0 - ouverture(i0, mat)

	return tmp

image = cv.imread(f"images/9x9test.png")
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
ret,image = cv.threshold(image,127,255, cv.THRESH_BINARY)

eros = erosion(image, boule3)
dila = dilatation(image, boule3)
ouve = ouverture(image, boule3)
ferm = fermeture(image, boule3)

fig = plt.figure(figsize = (9,6))
columns = 4
rows = 2

fig.add_subplot(rows, columns, 2)
plt.imshow(image, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('originale (9x9)')

fig.add_subplot(rows, columns, 3)
plt.imshow(boule3)
plt.axis('off')
plt.title('element structurant (3x3)')

fig.add_subplot(rows, columns, 5)
plt.imshow(dila, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('dilatation')

fig.add_subplot(rows, columns, 6)
plt.imshow(eros, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('erosion')

fig.add_subplot(rows, columns, 7)
plt.imshow(ouve, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('ouverture')

fig.add_subplot(rows, columns, 8)
plt.imshow(ferm, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('fermeture')

plt.show()

img = cv.imread(f"images/15x15test.png")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret,img = cv.threshold(img,127,255, cv.THRESH_BINARY)

skel = lantujoul(img, boule3, 10)

fig = plt.figure(figsize = (9,6))
columns = 2
rows = 1

fig.add_subplot(rows, columns, 1)
plt.imshow(img, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('originale')

fig.add_subplot(rows, columns, 2)
plt.imshow(skel, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('squelette')

plt.show()


image = cv.imread(f"images/boat.bmp")
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

eros = erosion(image, boule3)
dila = dilatation(image, boule3)
ouve = ouverture(image, boule3)
ferm = fermeture(image, boule3)

fig = plt.figure(figsize = (9,6))
columns = 4
rows = 2

fig.add_subplot(rows, columns, 2)
plt.imshow(image, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('originale (9x9)')

fig.add_subplot(rows, columns, 3)
plt.imshow(boule3)
plt.axis('off')
plt.title('element structurant (3x3)')

fig.add_subplot(rows, columns, 5)
plt.imshow(dila, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('dilatation')

fig.add_subplot(rows, columns, 6)
plt.imshow(eros, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('erosion')

fig.add_subplot(rows, columns, 7)
plt.imshow(ouve, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('ouverture')

fig.add_subplot(rows, columns, 8)
plt.imshow(ferm, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('fermeture')

plt.show()