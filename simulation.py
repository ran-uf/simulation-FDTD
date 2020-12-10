# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 10:20:05 2014

@author: Jukka Saarelma

"""
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import math
import time
from scipy.signal import chirp
from scipy import signal
import json
from samplesGenerate import samplesGenerate

# The FDTD library is loaded as module
import libPyFDTD as pf

###############################################################################
# Assign simulation parameters
###############################################################################

S_TYPE = int(sys.argv[1])
BOX = [int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])]

# dir = "./result/" + time.strftime("%Y-%m-%d-%H%M%S", time.localtime())

dir = "./result1/new_%d_%d_%d" % (BOX[0], BOX[1], BOX[2])
if not os.path.isdir(dir):
	os.mkdir(dir)
if S_TYPE == 0:
	dir = dir + "/sample"
elif S_TYPE == 1:
	dir = dir + "/ref"
elif S_TYPE == 2:
	dir = dir + "/no_sample"
os.mkdir(dir)
dir = dir + "/"
update_type = 0  # 0: SRL forward, 1: SRL sliced, 2: SRL centred
num_steps = 1000
fs = 10000  # 56230
double_precision = False
num_partition = 1

src_type = 0  # 0: Hard, 1: Soft, 2: Transparent
input_type = 3  # 0: Delta, 1: Gaussian, 2: Sine, 3: Given data
input_data_idx = 0

# delta + lowpass filter
# imp = signal.unit_impulse(fs/10)
imp = signal.unit_impulse(num_steps)
b, a = signal.butter(4, 0.2)
input_data = signal.lfilter(b, a, imp).tolist()
# plt.plot(input_data[0:250])
# plt.show()
angle_a = [0 / 180 * math.pi, 60 / 180 * math.pi, 120 / 180 * math.pi, 180 / 180 * math.pi, 240 / 180 * math.pi, 300 / 180 * math.pi]
angle_b = [30 / 180 * math.pi, 60 / 180 * math.pi]
srcs = []
r = 10
for a in angle_a:
	for b in angle_b:
		srcs.append([BOX[0] / 2 + r * math.cos(b) * math.sin(a), BOX[1] / 2 + r * math.cos(b) * math.cos(a), 10 + r * math.sin(b)])
srcs.append([BOX[0] / 2, BOX[1] / 2, 10 + r])

rec = []
rec_r = 5

for i in range(36):
	for j in range(36):
		rec.append([BOX[0] / 2 + rec_r * math.cos(math.pi / 72 * j) * math.cos(math.pi / 18 * i), BOX[1] / 2 + rec_r * math.cos(math.pi / 72 * j) * math.sin(math.pi / 18 * i), 10 + rec_r * math.sin(math.pi / 72 * j)])
rec.append([BOX[0] / 2, BOX[1] / 2, 10 + rec_r])

vertices, indices, mertials = samplesGenerate(dir, BOX, type=S_TYPE)
vertices = np.array(vertices)
indices = np.array(indices)
layer_list = mertials
layer_names = ['box', 'sample']
layers = {}

for k in range(0, len(layer_names)):
	layer_indices = [i for i, j in enumerate(layer_list) if j == layer_names[k]]
	layers[layer_names[k]] = layer_indices


# The solver takes admittance values
def reflection2Admittance(R):
	return (1.0 - R) / (1.0 + R)


def absorption2Admittance(alpha):
	return reflection2Admittance(np.sqrt(1.0 - alpha))


num_triangles = np.size(indices) // 3
num_coef = 20  # Default number of coefficients

# R_glob = 0.99
R_glob1 = 0.00
R_glob2 = 0.99
materials = np.ones((num_triangles, num_coef))  # *reflection2Admittance(R_glob)

materials[layers['box'], :] = reflection2Admittance(R_glob1)
materials[layers['sample'], :] = reflection2Admittance(R_glob2)

slice_n = [12, 14]
step = [400, 500]
orientation = [1, 1]
capture = [slice_n, step, orientation]

###############################################################################
# Initialize and run the FDTD solver
###############################################################################
for number, src in enumerate(srcs):

	app = pf.App()
	app.initializeDevices()
	app.initializeGeometryPy(indices.flatten().tolist(), vertices.flatten().tolist())
	app.setUpdateType(update_type)
	app.setNumSteps(int(num_steps))
	app.setSpatialFs(fs)
	app.setDouble(double_precision)
	app.forcePartitionTo(num_partition)
	app.addSurfaceMaterials(materials.flatten().tolist(), num_triangles, num_coef)
	# for s in srcs:
	# 	app.addSource(s[0], s[1], s[2], src_type, input_type, input_data_idx)
	app.addSource(src[0], src[1], src[2], src_type, input_type, input_data_idx)
	app.addSourceDataFloat(input_data, num_steps, 1)
	for i in range(0, np.shape(rec)[0]):
		app.addReceiver(rec[i][0], rec[i][1], rec[i][2])
	app.runSimulation()
	# app.runVisualization()
	ret = []
	for i in range(0, np.shape(rec)[0]):
		if double_precision:
			ret.append(np.array(app.getResponseDouble(i)))
		else:
			ret.append(np.array(app.getResponse(i)))
	np.save(dir + str(number) + "-output.npy", ret)
	del app

