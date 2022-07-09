import numpy as np 
import utils
from PIL import Image
import math
import random
import os 

src_dir=''
target_dir=''

class CameraModel():
	def __init__(self,param=[-0.3293,1.1258]):
		self.param=param

	def BTF(self,B0,k):
		beta=self.CRF(k)
		gamma=k**self.param[0]
		return beta*(B0**gamma)

	def CRF_Inv(self,B):
		E=(1-np.power(np.log(B)/self.param[1]),(1/self.param[0]))
		return E

	def CRF(self,E):
		return np.exp(self.param[1]*(1-E**self.param[0]))
	def adjust(self,im,k,round_n=False):
		k=np.tile(k,im.shape).astype(np.float32)
		out=self.BTF(im,k)
		out[out>1.0]=1.0
		if round_n:
			out=(((out*255).astype(np.uint8)).astype(np.float32))/255
		return out

if __name__=='__main__':
	model=CameraModel()
	for x in os.listdir(src_dir):
		im=utils.load_image(os.path.join(src_dir,x))
		k=random.uniform(1.8,2.4)
		out=model.adjust(im,k)
		out=out.transpose(1,2,0)
		out[out>1.]=1.
		out=Image.fromarray((out*255).astype(np.uint8))
		out.save(os.path.join(target_dir,x))
