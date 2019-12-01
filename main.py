import math
import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy

def contourVoting(luminances, normals, N):

	azimuth = np.array([math.atan2(y, z) if math.atan2(y, z) >= 0.0 else 2*math.pi+math.atan2(y, z) for (x, y, z) in normals])

	lumSort = np.array([(luminances[i], i) for i in range(len(luminances))])
	lumSort = np.sort(lumSort, axis=0)
	lumSort = lumSort[::-1]
	np.random.seed(5)
	phij = np.random.rand(N)*2*math.pi
	alphaAcc = np.zeros(N) # fix this variable
	kid = 0.5 			   # surface coefficient
	EPS = 1e-5
	running = True
	wj = np.array([0, math.sin(phij[0]), math.cos(phij[0])])
	while(running):
		for (_, i) in lumSort:
			i = int(i)
			omegaAcc = 0
			for j in range(N):
				wj = np.array([0, math.sin(phij[j]), math.cos(phij[j])])
				omegaAcc += kid*max(0, np.dot(normals[i], wj))
			running = False
			for j in range(N):
				wj = np.array([0, math.sin(phij[j]), math.cos(phij[j])])
				if(abs(omegaAcc) > EPS):
					alphaij = luminances[i]*max(0, np.dot(normals[i], wj))/omegaAcc
				else:
					alphaij = 5
				lastphij = phij[j]
				phij[j] = alphaAcc[j]*phij[j] + alphaij*azimuth[i]
				alphaAcc[j] = alphaAcc[j] + alphaij
				if(abs(alphaAcc[j]) > EPS):
					phij[j] = phij[j]/alphaAcc[j]
				else:
					phij[j] = 5.0
				if(abs(phij[j] - lastphij) > EPS):
					running = True
	return phij


def findZenithAngle(angle, img, h, w):
	pixel = [0, 0]
	if(angle > math.pi/4 and angle <= 3*math.pi/4):
		pixel[0] = int(round((angle-math.pi/4)/(3*math.pi/4-math.pi/4)*(h-1)))
		pixel[1] = w-1
	elif(angle > 3*math.pi/4 and angle <= 5*math.pi/4):
		pixel[0] = h-1
		pixel[1] = int(round((1 - ((angle-3*math.pi/4)/(5*math.pi/4-3*math.pi/4)))*(w-1)))
	elif(angle > 5*math.pi/4 and angle <= 7*math.pi/4):
		pixel[0] = int(round((1 - ((angle-5*math.pi/4)/(7*math.pi/4-5*math.pi/4)))*(h-1)))
		pixel[1] = 0
	else:
		newAngle = angle if angle < 0 else angle+2*math.pi
		pixel[0] = 0
		pixel[1] = int(round((newAngle-7*math.pi/4)/(9*math.pi/4-7*math.pi/4)*(w-1)))
	grayValue = -1
	newAngle = angle + math.pi/2 if angle + math.pi/2 <= 2*math.pi else angle + math.pi/2 - 2*math.pi
	du = math.sin(newAngle)
	dv = math.cos(newAngle)
	while(img[pixel[0]][pixel[1]] == 0):
		pixel = [round(pixel[0] + du), round(pixel[1] + dv)]
		if(pixel[0] < 0 or pixel[0] >= h or pixel[1] < 0 or pixel[1] >= w):
			break
	nextPixel = [round(pixel[0] + du), round(pixel[1] + dv)]
	zenith = math.pi/2
	if(nextPixel[0] >= 0 or nextPixel[0] < h or nextPixel[1] >= 0 or nextPixel[1] < w):
		derivative = int(int(img[nextPixel[0]][nextPixel[1]]) - int(img[pixel[0]][pixel[1]]))
		if(derivative > 0):			# if case 1
			while(derivative > 0):
				pixel = nextPixel
				nextPixel = [round(pixel[0] + du), round(pixel[1] + dv)]
				if(nextPixel[0] >= 0 or nextPixel[0] < h or nextPixel[1] >= 0 or nextPixel[1] < w):
					derivative = int(int(img[nextPixel[0]][nextPixel[1]]) - int(img[pixel[0]][pixel[1]]))
				else:
					derivative = -1
			zenith = math.pi/2        # should be the normal of the current pixel "pixel", but ...?
		elif (derivative < 0):					   # else case 2
			while(derivative < 0 or img[nextPixel[0]][nextPixel[1]] > 0):
				pixel = nextPixel
				nextPixel = [round(pixel[0] + du), round(pixel[1] + dv)]
				if(nextPixel[0] >= 0 or nextPixel[0] < h or nextPixel[1] >= 0 or nextPixel[1] < w):
					derivative = int(int(img[nextPixel[0]][nextPixel[1]]) - int(img[pixel[0]][pixel[1]]))
				else:
					derivative = 1
			zenith = 3*math.pi/2		 # should be the normal of the current pixel "pixel" but ...?
	return zenith



#realImg = cv2.imread('gato.jpg') #gato
realImg = cv2.imread('a.png')	 #leao
h, w = realImg.shape[:2]

gray = cv2.cvtColor(realImg, cv2.COLOR_BGR2GRAY)
a, newImg = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
newImg[:] = 0

#pt = [(177, 300)]
pt = [(116, 130)]
du = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
dv = [-1, -1, -1, 0, 0, 0, 1, 1, 1]

seen = np.zeros((h, w), dtype=np.int32)

edge = copy.deepcopy(newImg)

for (x, y) in pt:
	queue = [(x, y)]
	while(len(queue) > 0):
		(u, v) = queue[0]
		queue.pop(0)
		if(seen[u][v]):
			continue
		seen[u][v] = 1
		newImg[u][v] = 255
		for i in range(len(du)):
			nu = u + du[i]
			nv = v + dv[i]
			if(nu < 0 or nu >= h or nv < 0 or nv >= w):
				continue
			if(seen[nu][nv] == 0 and gray[nu][nv] > 0):
				queue.append((nu, nv))
			elif(gray[nu][nv] == 0):
				edge[u][v] = 255

normals = []

dict = {
	(0, 1, 0, 0, 1, 0, 0, 0, 0): (0, 0, -1),
	(0, 0, 0, 1, 1, 0, 0, 0, 0): (0, 1, 0),
	(0, 0, 0, 0, 1, 1, 0, 0, 0): (0, -1, 0),
	(0, 0, 0, 0, 1, 0, 0, 1, 0): (0, 0, 1),
	(1, 0, 0, 0, 1, 0, 0, 0, 0): (0, 1, -1),
	(0, 0, 1, 0, 1, 0, 0, 0, 0): (0, -1, -1),
	(0, 0, 0, 0, 1, 0, 1, 0, 0): (0, 1, 1),
	(0, 0, 0, 0, 1, 0, 0, 0, 1): (0, -1, 1),
	(1, 1, 0, 1, 1, 0, 0, 0, 0): (0, 1, -1),
	(0, 1, 1, 0, 1, 1, 0, 0, 0): (0, -1, -1),
	(0, 0, 0, 1, 1, 0, 1, 1, 0): (0, 1, 1),
	(0, 0, 0, 0, 1, 1, 0, 1, 1): (0, -1, 1),
	(0, 0, 1, 0, 1, 1, 0, 1, 1): (0, -1, 1),
	(0, 1, 1, 0, 1, 1, 0, 0, 1): (0, -1, -1),
	(0, 0, 1, 0, 1, 1, 1, 1, 1): (0, -1, 1),
	(0, 1, 1, 1, 1, 1, 1, 1, 1): (0, -1, 1),
	(1, 1, 1, 1, 1, 1, 0, 1, 1): (0, -1, -1),
	(1, 1, 1, 0, 1, 1, 0, 1, 1): (0, -1, 0),
	(0, 1, 1, 0, 1, 1, 0, 1, 1): (0, -1, 0),
	(0, 1, 1, 0, 1, 1, 1, 1, 1): (0, -1, 0),
	(1, 1, 1, 0, 1, 1, 0, 0, 1): (0, -1, 1),
	(0, 0, 0, 0, 1, 1, 1, 1, 1): (0, -1, 1),
	(0, 0, 1, 1, 1, 1, 1, 1, 1): (0, 0, 1),
	(1, 1, 1, 0, 1, 1, 0, 0, 0): (0, -1, -1),
	(1, 1, 1, 1, 1, 1, 0, 0, 0): (0, 0, -1),
	(0, 0, 0, 1, 1, 1, 1, 1, 1): (0, 0, 1),
	(1, 1, 1, 1, 1, 1, 0, 0, 1): (0, 0, -1),
	(1, 1, 1, 1, 1, 1, 1, 1, 0): (0, 1, -1),
	(1, 1, 1, 1, 1, 0, 0, 0, 0): (0, 1, -1),
	(0, 0, 0, 1, 1, 0, 1, 1, 1): (0, 1, 1),
	(1, 1, 0, 1, 1, 1, 1, 1, 1): (0, 1, -1),
	(1, 1, 1, 1, 1, 1, 1, 0, 0): (0, 0, -1),
	(1, 0, 0, 1, 1, 0, 1, 1, 0): (0, 1, 1),
	(1, 1, 0, 1, 1, 0, 1, 1, 1): (0, 1, 0),
	(1, 0, 0, 1, 1, 0, 1, 1, 1): (0, 1, 1),
	(1, 0, 0, 1, 1, 1, 1, 1, 1): (0, 0, 1),
	(1, 0, 1, 1, 1, 1, 1, 1, 1): (0, 0, 1),
	(1, 1, 1, 1, 1, 1, 0, 1, 0): (0, 0, -1),
	(1, 1, 1, 0, 1, 0, 0, 0, 0): (0, 0, -1),
	(1, 1, 1, 1, 1, 0, 1, 0, 0): (0, -1, -1),
	(1, 1, 1, 1, 1, 1, 1, 0, 1): (0, 0, -1),
	(1, 1, 0, 1, 1, 0, 1, 1, 0): (0, 1, 0),
	(1, 1, 1, 1, 1, 0, 1, 1, 0): (0, 1, 0),
	(1, 1, 0, 1, 1, 0, 1, 0, 0): (0, 1, 0),
	(0, 0, 1, 0, 1, 1, 0, 0, 1): (0, -1, 0)
}
lum = []

for u in range(h):
	for v in range(w):
		if(edge[u][v] > 0):
			number = [0]*9
			for i in range(len(du)):
				nu = u + du[i]
				nv = v + dv[i]
				if(nu > 0 and nu < h and nv > 0 and nv < w):
					number[i] = 1 if newImg[nu][nv] > 0 else 0
			lum.append(realImg[u][v][0]*0.11 + realImg[u][v][1]*0.59 + realImg[u][v][2]*0.3)
			if(tuple(number) in dict.keys()):
				normals.append((dict[tuple(number)])/np.linalg.norm(dict[tuple(number)]))
			else:
				alpha = np.random.rand(1)*2*math.pi
				normals.append((0, math.cos(alpha), math.sin(alpha)))

lum = np.array(lum)
normals = np.array(normals)
omegaj = contourVoting(lum, normals, 3)
andImg = copy.deepcopy(newImg[:])
andImg[andImg > 0] = 1
andImg = gray*andImg

zenithAngles = []

for azimuth in omegaj:
	zenith = findZenithAngle(azimuth, andImg, h, w)
	zenithAngles.append(zenith)

omegaAngles = [180.0/math.pi*x for x in omegaj]

print(omegaAngles, zenithAngles)

#plt.imshow(realImg), plt.show()
cv2.imshow("name", andImg), cv2.waitKey(0), cv2.destroyAllWindows()
