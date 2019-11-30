import math
import numpy as np
import cv2
from matplotlib import pyplot as plt

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

img = cv2.imread('test.png')
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

h, w= img.shape[:2]

rect = (0,0,w-1,h-1)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img),plt.colorbar(),plt.show()