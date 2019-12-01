import math
import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy

def contourVoting(luminances, normals, N):
	azimuth = np.array([math.atan2(y, z) if math.atan2(y, z) >= 0.0 else 2*math.pi+math.atan2(y, z) for (x, y, z) in normals])
	lumSort = np.array([(luminances[i], i) for i in range(len(luminances))])
	lumSort = np.sort(lumSort, axis=0)
	phij = np.random.rand(N)*360.0
	alphaAcc = np.zeros(N) # fix this variable
	kid = 0.5  			   # surface coefficient
	EPS = 1e-6
	running = True
	while(running):
		for (_, i) in lumSort:
			omegaAcc = 0
			for j in range(N):
				wj = np.tranpose(np.array([0, sin(phij[j]), cos(phij[j])]))
				omegaAcc += kid*max(0, np.dot(normals[i], wj))
			running = False
			for j in range(N):
				wj = np.tranpose(np.array([0, sin(phij[j]), cos(phij[j])]))
				alphaij = luminances[i]*max(0, np.dot(normals[i], wj))/omegaAcc
				lastphij = phij[j]
				phij[j] = alphaAcc[j]*phij[j] + alphaij*azimuth[i]
				alphaAcc[j] = alphaAcc[j] + alphaij
				phij[j] = phij[j]/alphaAcc[j]
				if(abs(phij[j] - lastphij) > EPS):
					running = True
	return phij;

realImg = cv2.imread('a.png')
h, w = realImg.shape[:2]
lum = np.array([pixel[0]*0.3 + pixel[1]*0.59 + pixel[2]*0.11 for row in realImg for pixel in row])
'''img = cv2.imread('a.png')
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (0,0,w-1,h-1)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]'''

gray = cv2.cvtColor(realImg, cv2.COLOR_BGR2GRAY)
a, gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
newImg = copy.deepcopy(gray)
newImg[:] = 0


pt = [(116, 130), (59, 84)]
du = [0, 0, 1, 1, 1, -1, -1, -1]
dv = [1, -1, 0, 1, -1, 0, 1, -1]

seen = np.zeros((h, w), dtype=np.int32)

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
			if(seen[nu][nv] == 0 and gray[nu][nv] > 0):
				queue.append((nu, nv))
	

newImg = cv2.Canny(newImg,100,200)

cv2.imshow("Converted Image",newImg)

# waiting for key event
cv2.waitKey(0)

# destroying all windows
cv2.destroyAllWindows()