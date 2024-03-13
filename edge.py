import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def Maxima_loco(src, bsize = 3, ksize = 3):

	img = cv.imread(src)
	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img = cv.GaussianBlur(img, (bsize,bsize), 0)


	# Apply Sobelx in high output datatype 'float32'
	# and then converting back to 8-bit to prevent overflow
	sobelx_64 = cv.Sobel(img,cv.CV_32F,1,0,ksize=ksize)
	absx_64 = np.absolute(sobelx_64)
	sobelx_8u1 = absx_64/absx_64.max()*255
	sobelx_8u = np.uint8(sobelx_8u1)
	 
	# Similarly for Sobely
	sobely_64 = cv.Sobel(img,cv.CV_32F,0,1,ksize=ksize)
	absy_64 = np.absolute(sobely_64)
	sobely_8u1 = absy_64/absy_64.max()*255
	sobely_8u = np.uint8(sobely_8u1)
	 
	# From gradients calculate the magnitude and changing
	# it to 8-bit (Optional)
	mag = np.hypot(sobelx_8u, sobely_8u)
	mag = mag/mag.max()*255
	mag = np.uint8(mag)
	 
	# Find the direction and change it to degree
	theta = np.arctan2(sobely_64, sobelx_64)
	angle = np.rad2deg(theta)

	# Find the neighbouring pixels (b,c) in the rounded gradient direction
	# and then apply non-max suppression
	M, N = mag.shape
	Non_max = np.zeros((M,N), dtype= np.uint8)

	for i in range(1,M-1):
	    for j in range(1,N-1):
	       # Horizontal 0
	        if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180) or (-22.5 <= angle[i,j] < 0) or (-180 <= angle[i,j] < -157.5):
	            b = mag[i, j+1]
	            c = mag[i, j-1]
	        # Diagonal 45
	        elif (22.5 <= angle[i,j] < 67.5) or (-157.5 <= angle[i,j] < -112.5):
	            b = mag[i+1, j+1]
	            c = mag[i-1, j-1]
	        # Vertical 90
	        elif (67.5 <= angle[i,j] < 112.5) or (-112.5 <= angle[i,j] < -67.5):
	            b = mag[i+1, j]
	            c = mag[i-1, j]
	        # Diagonal 135
	        elif (112.5 <= angle[i,j] < 157.5) or (-67.5 <= angle[i,j] < -22.5):
	            b = mag[i+1, j-1]
	            c = mag[i-1, j+1]           
	            
	        # Non-max Suppression
	        if (mag[i,j] >= b) and (mag[i,j] >= c):
	            Non_max[i,j] = mag[i,j]
	        else:
	            Non_max[i,j] = 0

	return Non_max

def Hysteresis_8x(src, sb = 50, sh = 150, bsize = 3, ksize = 3):

	img = cv.imread(src)
	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img = cv.GaussianBlur(img, (bsize,bsize), 0)


	# Apply Sobelx in high output datatype 'float32'
	# and then converting back to 8-bit to prevent overflow
	sobelx_64 = cv.Sobel(img,cv.CV_32F,1,0,ksize=ksize)
	absx_64 = np.absolute(sobelx_64)
	sobelx_8u1 = absx_64/absx_64.max()*255
	sobelx_8u = np.uint8(sobelx_8u1)
	 
	# Similarly for Sobely
	sobely_64 = cv.Sobel(img,cv.CV_32F,0,1,ksize=ksize)
	absy_64 = np.absolute(sobely_64)
	sobely_8u1 = absy_64/absy_64.max()*255
	sobely_8u = np.uint8(sobely_8u1)
	 
	# From gradients calculate the magnitude and changing
	# it to 8-bit (Optional)
	mag = np.hypot(sobelx_8u, sobely_8u)
	mag = mag/mag.max()*255
	mag = np.uint8(mag)
	 
	# Find the direction and change it to degree
	theta = np.arctan2(sobely_64, sobelx_64)
	angle = np.rad2deg(theta)

	# Find the neighbouring pixels (b,c) in the rounded gradient direction
	# and then apply non-max suppression
	M, N = mag.shape
	Non_max = np.zeros((M,N), dtype= np.uint8)

	for i in range(1,M-1):
	    for j in range(1,N-1):
	       # Horizontal 0
	        if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180) or (-22.5 <= angle[i,j] < 0) or (-180 <= angle[i,j] < -157.5):
	            b = mag[i, j+1]
	            c = mag[i, j-1]
	        # Diagonal 45
	        elif (22.5 <= angle[i,j] < 67.5) or (-157.5 <= angle[i,j] < -112.5):
	            b = mag[i+1, j+1]
	            c = mag[i-1, j-1]
	        # Vertical 90
	        elif (67.5 <= angle[i,j] < 112.5) or (-112.5 <= angle[i,j] < -67.5):
	            b = mag[i+1, j]
	            c = mag[i-1, j]
	        # Diagonal 135
	        elif (112.5 <= angle[i,j] < 157.5) or (-67.5 <= angle[i,j] < -22.5):
	            b = mag[i+1, j-1]
	            c = mag[i-1, j+1]           
	            
	        # Non-max Suppression
	        if (mag[i,j] >= b) and (mag[i,j] >= c):
	            Non_max[i,j] = mag[i,j]
	        else:
	            Non_max[i,j] = 0

	# Set high and low threshold
	highThreshold = sh
	lowThreshold = sb

	M, N = Non_max.shape
	out = np.zeros((M,N), dtype= np.uint8)

	# If edge intensity is greater than 'High' it is a sure-edge
	# below 'low' threshold, it is a sure non-edge
	strong_i, strong_j = np.where(Non_max >= highThreshold)
	zeros_i, zeros_j = np.where(Non_max < lowThreshold)

	# weak edges
	weak_i, weak_j = np.where((Non_max <= highThreshold) & (Non_max >= lowThreshold))

	# Set same intensity value for all edge pixels
	out[strong_i, strong_j] = 255
	out[zeros_i, zeros_j ] = 0
	out[weak_i, weak_j] = 75

	M, N = out.shape
	for i in range(1, M-1):
	    for j in range(1, N-1):
	        if (out[i,j] == 75):
	            if 255 in [out[i+1, j-1],out[i+1, j],out[i+1, j+1],out[i, j-1],out[i, j+1],out[i-1, j-1],out[i-1, j],out[i-1, j+1]]:
	                out[i, j] = 255
	            else:
	                out[i, j] = 0

	return out

def Hysteresis_8x_v2(src, sb = 50, sh = 150, bsize = 3, ksize = 3):

	img = cv.imread(src)
	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img = cv.GaussianBlur(img, (bsize,bsize), 0)


	# Apply Sobelx in high output datatype 'float32'
	# and then converting back to 8-bit to prevent overflow
	sobelx_64 = cv.Sobel(img,cv.CV_32F,1,0,ksize=ksize)
	absx_64 = np.absolute(sobelx_64)
	sobelx_8u1 = absx_64/absx_64.max()*255
	sobelx_8u = np.uint8(sobelx_8u1)
	 
	# Similarly for Sobely
	sobely_64 = cv.Sobel(img,cv.CV_32F,0,1,ksize=ksize)
	absy_64 = np.absolute(sobely_64)
	sobely_8u1 = absy_64/absy_64.max()*255
	sobely_8u = np.uint8(sobely_8u1)
	 
	# From gradients calculate the magnitude and changing
	# it to 8-bit (Optional)
	mag = np.hypot(sobelx_8u, sobely_8u)
	mag = mag/mag.max()*255
	mag = np.uint8(mag)

	# Set high and low threshold
	highThreshold = sh
	lowThreshold = sb

	M, N = img.shape
	out = np.zeros((M,N), dtype= np.uint8)

	# If edge intensity is greater than 'High' it is a sure-edge
	# below 'low' threshold, it is a sure non-edge
	strong_i, strong_j = np.where(mag >= highThreshold)
	zeros_i, zeros_j = np.where(mag < lowThreshold)

	# weak edges
	weak_i, weak_j = np.where((mag <= highThreshold) & (mag >= lowThreshold))

	# Set same intensity value for all edge pixels
	out[strong_i, strong_j] = 255
	out[zeros_i, zeros_j ] = 0
	out[weak_i, weak_j] = 75

	M, N = out.shape
	for i in range(1, M-1):
	    for j in range(1, N-1):
	        if (out[i,j] == 75):
	            if 255 in [out[i+1, j-1],out[i+1, j],out[i+1, j+1],out[i, j-1],out[i, j+1],out[i-1, j-1],out[i-1, j],out[i-1, j+1]]:
	                out[i, j] = 255
	            else:
	                out[i, j] = 0

	return out

def Hysteresis_4x(src, sb = 50, sh = 150, bsize = 3, ksize = 3):

	img = cv.imread(src)
	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img = cv.GaussianBlur(img, (bsize,bsize), 0)

	# Apply Sobelx in high output datatype 'float32'
	# and then converting back to 8-bit to prevent overflow
	filter_x = np.matrix([[0,0,0],[-1,0,1],[0,0,0]], np.int8)
	sobelx_64 = cv.filter2D(img, cv.CV_32F, filter_x)
	absx_64 = np.absolute(sobelx_64)
	sobelx_8u1 = absx_64/absx_64.max()*255
	sobelx_8u = np.uint8(sobelx_8u1)
	 
	# Similarly for Sobely
	filter_y = np.matrix([[0,-1,0],[0,0,0],[0,1,0]], np.int8)
	sobely_64 = cv.filter2D(img, cv.CV_32F, filter_y)
	absy_64 = np.absolute(sobely_64)
	sobely_8u1 = absy_64/absy_64.max()*255
	sobely_8u = np.uint8(sobely_8u1)
	 
	# From gradients calculate the magnitude and changing
	# it to 8-bit (Optional)
	mag = np.hypot(sobelx_8u, sobely_8u)
	mag = mag/mag.max()*255
	mag = np.uint8(mag)
	 
	# Find the direction and change it to degree
	theta = np.arctan2(sobely_64, sobelx_64)
	angle = np.rad2deg(theta)

	# Find the neighbouring pixels (b,c) in the rounded gradient direction
	# and then apply non-max suppression
	M, N = mag.shape
	Non_max = np.zeros((M,N), dtype= np.uint8)

	for i in range(1,M-1):
	    for j in range(1,N-1):
	       # Horizontal 0
	        if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180) or (-22.5 <= angle[i,j] < 0) or (-180 <= angle[i,j] < -157.5):
	            b = mag[i, j+1]
	            c = mag[i, j-1]
	        # Diagonal 45
	        elif (22.5 <= angle[i,j] < 67.5) or (-157.5 <= angle[i,j] < -112.5):
	            b = mag[i+1, j+1]
	            c = mag[i-1, j-1]
	        # Vertical 90
	        elif (67.5 <= angle[i,j] < 112.5) or (-112.5 <= angle[i,j] < -67.5):
	            b = mag[i+1, j]
	            c = mag[i-1, j]
	        # Diagonal 135
	        elif (112.5 <= angle[i,j] < 157.5) or (-67.5 <= angle[i,j] < -22.5):
	            b = mag[i+1, j-1]
	            c = mag[i-1, j+1]           
	            
	        # Non-max Suppression
	        if (mag[i,j] >= b) and (mag[i,j] >= c):
	            Non_max[i,j] = mag[i,j]
	        else:
	            Non_max[i,j] = 0

	# Set high and low threshold
	highThreshold = sh
	lowThreshold = sb

	M, N = Non_max.shape
	out = np.zeros((M,N), dtype= np.uint8)

	# If edge intensity is greater than 'High' it is a sure-edge
	# below 'low' threshold, it is a sure non-edge
	strong_i, strong_j = np.where(Non_max >= highThreshold)
	zeros_i, zeros_j = np.where(Non_max < lowThreshold)

	# weak edges
	weak_i, weak_j = np.where((Non_max <= highThreshold) & (Non_max >= lowThreshold))

	# Set same intensity value for all edge pixels
	out[strong_i, strong_j] = 255
	out[zeros_i, zeros_j ] = 0
	out[weak_i, weak_j] = 75

	M, N = out.shape
	for i in range(1, M-1):
	    for j in range(1, N-1):
	        if (out[i,j] == 75):
	            if 255 in [out[i+1, j-1],out[i+1, j],out[i+1, j+1],out[i, j-1],out[i, j+1],out[i-1, j-1],out[i-1, j],out[i-1, j+1]]:
	                out[i, j] = 255
	            else:
	                out[i, j] = 0

	return out

def Hysteresis_4x_v2(src, sb = 50, sh = 150, bsize = 3, ksize = 3):

	img = cv.imread(src)
	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img = cv.GaussianBlur(img, (bsize,bsize), 0)

	# Apply Sobelx in high output datatype 'float32'
	# and then converting back to 8-bit to prevent overflow
	filter_x = np.matrix([[0,0,0],[-1,0,1],[0,0,0]], np.int8)
	sobelx_64 = cv.filter2D(img, cv.CV_32F, filter_x)
	absx_64 = np.absolute(sobelx_64)
	sobelx_8u1 = absx_64/absx_64.max()*255
	sobelx_8u = np.uint8(sobelx_8u1)
	 
	# Similarly for Sobely
	filter_y = np.matrix([[0,-1,0],[0,0,0],[0,1,0]], np.int8)
	sobely_64 = cv.filter2D(img, cv.CV_32F, filter_y)
	absy_64 = np.absolute(sobely_64)
	sobely_8u1 = absy_64/absy_64.max()*255
	sobely_8u = np.uint8(sobely_8u1)
	 
	# From gradients calculate the magnitude and changing
	# it to 8-bit (Optional)
	mag = np.hypot(sobelx_8u, sobely_8u)
	mag = mag/mag.max()*255
	mag = np.uint8(mag)

	# Find the neighbouring pixels (b,c) in the rounded gradient direction
	# and then apply non-max suppression
	M, N = mag.shape

	# Set high and low threshold
	highThreshold = sh
	lowThreshold = sb

	out = np.zeros((M,N), dtype= np.uint8)

	# If edge intensity is greater than 'High' it is a sure-edge
	# below 'low' threshold, it is a sure non-edge
	strong_i, strong_j = np.where(mag >= highThreshold)
	zeros_i, zeros_j = np.where(mag < lowThreshold)

	# weak edges
	weak_i, weak_j = np.where((mag <= highThreshold) & (mag >= lowThreshold))

	# Set same intensity value for all edge pixels
	out[strong_i, strong_j] = 255
	out[zeros_i, zeros_j ] = 0
	out[weak_i, weak_j] = 75

	M, N = out.shape
	for i in range(1, M-1):
	    for j in range(1, N-1):
	        if (out[i,j] == 75):
	            if 255 in [out[i+1, j-1],out[i+1, j],out[i+1, j+1],out[i, j-1],out[i, j+1],out[i-1, j-1],out[i-1, j],out[i-1, j+1]]:
	                out[i, j] = 255
	            else:
	                out[i, j] = 0

	return out

def Sobel(src, ksize = 3, scale = 1, delta = 0, ddepth = cv.CV_16S, borderType = cv.BORDER_DEFAULT):

	img = cv.imread(src)

	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img = cv.GaussianBlur(img, (3,3), 0)

	Gx = cv.Sobel(img, ddepth, 1, 0, ksize = ksize, scale = scale, delta = delta, borderType = borderType)
	Gy = cv.Sobel(img, ddepth, 0, 1, ksize = ksize, scale = scale, delta = delta, borderType = borderType)

	Abs_Gx = cv.convertScaleAbs(Gx)
	Abs_Gy = cv.convertScaleAbs(Gy)

	Gr = cv.addWeighted(Abs_Gx, 0.5, Abs_Gy, 0.5, 0)
	return Gr

def Sobel_4x(src):

	img = cv.imread(src)

	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img = cv.GaussianBlur(img, (3,3), 0)

	filter_x = np.matrix([[0,0,0],[-1,0,1],[0,0,0]], np.int8)
	filter_y = np.matrix([[0,-1,0],[0,0,0],[0,1,0]], np.int8)

	Gx = cv.filter2D(img, cv.CV_32F, filter_x)
	Gx = np.absolute(Gx)
	Gx = Gx/Gx.max()*255
	Gx = np.uint8(Gx)

	Gy = cv.filter2D(img, cv.CV_32F, filter_y)
	Gy = np.absolute(Gy)
	Gy = Gy/Gy.max()*255
	Gy = np.uint8(Gy)

	Abs_Gx = cv.convertScaleAbs(Gx)
	Abs_Gy = cv.convertScaleAbs(Gy)

	Gr = cv.addWeighted(Abs_Gx, 0.5, Abs_Gy, 0.5, 0)
	return Gr

def Sobel_8x(src):

	img = cv.imread(src)

	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img = cv.GaussianBlur(img, (3,3), 0)

	filter_x = np.matrix([[-1,0,1],[-2,0,2],[-1,0,1]], np.int8)
	filter_y = np.matrix([[-1,-2,-1],[0,0,0],[1,2,1]], np.int8)

	Gx = cv.filter2D(img, cv.CV_32F, filter_x)
	Gx = np.absolute(Gx)
	Gx = Gx/Gx.max()*255
	Gx = np.uint8(Gx)

	Gy = cv.filter2D(img, cv.CV_32F, filter_y)
	Gy = np.absolute(Gy)
	Gy = Gy/Gy.max()*255
	Gy = np.uint8(Gy)

	Abs_Gx = cv.convertScaleAbs(Gx)
	Abs_Gy = cv.convertScaleAbs(Gy)

	Gr = cv.addWeighted(Abs_Gx, 0.5, Abs_Gy, 0.5, 0)
	return Gr

def Laplace(src, ksize = 3, ddepth = cv.CV_16S):

	img = cv.imread(src)

	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img = cv.GaussianBlur(img, (3,3), 0)

	result = cv.Laplacian(img, ddepth, ksize = ksize)
	result = cv.convertScaleAbs(result)
	
	return result

source = "images/boat.bmp"

sobel = Sobel(source)
sobel_4x = Sobel(source)
sobel_8x = Sobel_8x(source)
laplace = Laplace(source, ksize = 3)
hyster8  = Hysteresis_8x(source, 40, 70, bsize = 3, ksize = 3)
hyster8_2  = Hysteresis_8x_v2(source, 40, 70, bsize = 3, ksize = 3)
hyster4  = Hysteresis_4x(source, 40, 70, bsize = 3, ksize = 3)
hyster4_2  = Hysteresis_4x_v2(source, 40, 70, bsize = 3, ksize = 3)
maxima = Maxima_loco(source, bsize = 3, ksize = 3)

diff = hyster8 - hyster4
diff = cv.convertScaleAbs(diff)

cv.imwrite("images/output/laplace.png", laplace)

cv.imwrite("images/output/sobel.png", sobel)
cv.imwrite("images/output/sobel_4x.png", sobel_4x)
cv.imwrite("images/output/sobel_8x.png", sobel_8x)

cv.imwrite("images/output/hysteresis_8x.png", hyster8)
cv.imwrite("images/output/hysteresis_8x_2.png", hyster8_2)
cv.imwrite("images/output/hysteresis_4x.png", hyster4)
cv.imwrite("images/output/hysteresis_4x_2.png", hyster4_2)

cv.imwrite("images/output/diff.png", diff)
cv.imwrite("images/output/maxim.png", maxima)

fig = plt.figure(figsize = (14,8))
columns = 4
rows = 2

fig.add_subplot(rows, columns, 1)
plt.imshow(hyster4, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('Hysteresis_4x maxima locaux')

fig.add_subplot(rows, columns, 2)
plt.imshow(hyster4_2, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('Hysteresis_4x norme gradient')

fig.add_subplot(rows, columns, 3)
plt.imshow(hyster8, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('Hysteresis_8x maxima locaux')

fig.add_subplot(rows, columns, 4)
plt.imshow(hyster8_2, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('Hysteresis_8x norme gradient')


fig.add_subplot(rows, columns, 5)
plt.imshow(sobel, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('Sobel (openCv)')

fig.add_subplot(rows, columns, 6)
plt.imshow(sobel_4x, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('Sobel 4x')

fig.add_subplot(rows, columns, 7)
plt.imshow(sobel_8x, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('Sobel_8x')

fig.add_subplot(rows, columns, 8)
plt.imshow(laplace, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('Laplace')



plt.show()
