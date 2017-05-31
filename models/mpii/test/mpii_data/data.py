import logging
import random as rand
from enum import Enum

import numpy as np
from numpy import array as arr
from numpy import concatenate as cat

import scipy as sy
import scipy.io as sio
from scipy.misc import imread, imresize

pred = sio.loadmat('predictions.mat')
pred = pred['joints']
mlab = sio.loadmat('dataset.mat')
mlab = mlab['dataset']

#print(pred[0,0])
#print(pred[0,0][:,1])
#print(mlab[0,0][2][0])
#print(type(mlab[0,0][2][0]))
a = mlab[0,0][2][0][0]
num_images = mlab.shape[1]
data = []
for i in range(num_images):
  ele = mlab[0,i][2][0][0] #array with joint data
  data.append(ele)

#print(len(data))
#compute distance between points
dist = np.zeros(num_images)
for i in range(num_images):
  #x-data
  img = data[i]
  act_x = img[:,1]
  act_y = img[:,2]
  #print(act_y.shape)

  pred_img = pred[0,i]
  pred_x = pred_img[:,0]
  pred_y = pred_img[:,1]
  #print(pred_y.shape)
  pred_weight = pred_img[:,2]

  if(len(act_y)==14):
    dist[i] += sy.spatial.distance.euclidean(act_x,pred_x)

eucl = []
for i in range(num_images):
  if dist[i]!= 0.0:
    eucl.append(dist[i])
print(eucl)
  


