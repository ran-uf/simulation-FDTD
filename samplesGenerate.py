# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 9:10:09 2020

@author: Jukka Saarelma, Ran Dou

"""
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mp
from PIL import Image
from PIL import ImageFilter
from skimage import filters
import json

SAMPLERATE = 50


def index(x, y):
    return y * SAMPLERATE + x


def save_obj(vertices, indices, dir):
    with open(dir + 'sample.obj', "w") as m:
        m.writelines("g\n")
        for v in range(len(vertices)):
            m.writelines("v {} {} {}\n".format(vertices[v][0], vertices[v][1], vertices[v][2]))
        for i in range(len(indices)):
            m.writelines("f {} {} {}\n".format(indices[i][0] + 1, indices[i][1] + 1, indices[i][2] + 1))
        m.writelines("\n")
        m.writelines("\n")
        m.close()


# x,y plane, z height
def samplesGenerate(dir, BOX, type=0):
    vertices = []
    indices = []
    mertials = []
    pos = [BOX[0] / 2 - 0.5, BOX[1] / 2 - 0.5, 10]
    img = None
    s = - 4
    if type == 0:
        numpy.random.seed(0)
    #   zs = numpy.ones(SAMPLERATE ** 2).tolist()
        zs = []
        for i in range(SAMPLERATE):
            for j in range(SAMPLERATE):
                zs.append((0.45 * numpy.sin(j / 10 * 3.14)) ** 2 + 0.05)
        '''''''''''''''''''''random'''''''''''''''''''''
        # zs = numpy.random.random(SAMPLERATE ** 2) * 0.45 + 0.05
        img = numpy.reshape(zs, [SAMPLERATE, SAMPLERATE])
        # img = filters.gaussian(img, sigma=1.5)
        mp.imsave(dir + "image.png", img)
        numpy.save(dir + "input.npy", img)
        zs = img.reshape(-1).tolist()
    elif type == 1:
        zs = numpy.zeros(SAMPLERATE ** 2) + 0.05
        img = numpy.reshape(zs, [SAMPLERATE, SAMPLERATE])
        mp.imsave(dir + "image.png", img)
        numpy.save(dir + "input.npy", img)
        zs = img.reshape(-1).tolist()

    if type != 2:
        for i in range(SAMPLERATE):
            for j in range(SAMPLERATE):
                vertices.append([i / SAMPLERATE + pos[0], j / SAMPLERATE + pos[1], zs[index(i, j)] + pos[2]])
        vertices.append([pos[0], pos[1], pos[2]])
        vertices.append([pos[0], pos[1] + 1 - 1 / SAMPLERATE, pos[2]])
        vertices.append([pos[0] + 1 - 1 / SAMPLERATE, pos[1], pos[2]])
        vertices.append([pos[0] + 1 - 1 / SAMPLERATE, pos[1] + 1 - 1 / SAMPLERATE, pos[2]])

        for i in range(SAMPLERATE - 1):
            for j in range(SAMPLERATE - 1):
                indices.append([index(i, j), index(i + 1, j), index(i, j + 1)])
                indices.append([index(i + 1, j), index(i + 1, j + 1), index(i, j + 1)])
                mertials.append('sample')
                mertials.append('sample')

        s = SAMPLERATE ** 2
        indices.append([s + 1, s + 2, s + 3])
        indices.append([s, s + 2, s + 1])
        mertials.append('sample')
        mertials.append('sample')

        # bound meshes
        for i in range(SAMPLERATE - 1):
            indices.append([s, i + 1, i])
            mertials.append('sample')
        indices.append([s, s + 1,  SAMPLERATE - 1])
        mertials.append('sample')

        for i in range(SAMPLERATE - 1):
            indices.append([index(0, i), index(0, i + 1), s])
            mertials.append('sample')
        indices.append([s, index(0, SAMPLERATE - 1), s + 2])
        mertials.append('sample')
    
        for i in range(SAMPLERATE - 1):
            indices.append([s + 1, index(SAMPLERATE - 1, i + 1), index(SAMPLERATE - 1, i)])
            mertials.append('sample')
        indices.append([s + 1, s + 3, SAMPLERATE ** 2 - 1])
        mertials.append('sample')

        for i in range(SAMPLERATE - 1):
            indices.append([s + 2, index(i, SAMPLERATE - 1), index(i + 1, SAMPLERATE - 1)])
            mertials.append('sample')
        indices.append([s + 2, SAMPLERATE ** 2 - 1, s + 3])
        mertials.append('sample')

    vertices.append([0,           0,      0])      # s + 4
    vertices.append([BOX[0],      0,      0])      # s + 5
    vertices.append([BOX[0], BOX[1],      0])      # s + 6
    vertices.append([0,      BOX[1],      0])      # s + 7
    vertices.append([0,           0, BOX[2]])
    vertices.append([BOX[0],      0, BOX[2]])
    vertices.append([BOX[0], BOX[1], BOX[2]])
    vertices.append([0,      BOX[1], BOX[2]])
    indices.append([s + 4, s + 6, s + 5])
    indices.append([s + 4, s + 7, s + 6])
    indices.append([s + 5, s + 9, s + 8])
    indices.append([s + 5, s + 8, s + 4])
    indices.append([s + 5, s + 6, s + 10])
    indices.append([s + 5, s + 10, s + 9])
    indices.append([s + 4, s + 8, s + 11])
    indices.append([s + 4, s + 11, s + 7])
    indices.append([s + 6, s + 7, s + 11])
    indices.append([s + 6, s + 11, s + 10])
    indices.append([s + 8, s + 9, s + 10])
    indices.append([s + 8, s + 10, s + 11])
    mertials.append('box')
    mertials.append('box')
    mertials.append('box')
    mertials.append('box')
    mertials.append('box')
    mertials.append('box')
    mertials.append('box')
    mertials.append('box')
    mertials.append('box')
    mertials.append('box')
    mertials.append('box')
    mertials.append('box')

    save_obj(vertices, indices, dir)
    return vertices, indices, mertials


if __name__ == "__main__":

    '''
    with open('sample.obj', "w") as m:
        m.writelines("g\n")
        for v in range(len(vertices)):
            m.writelines("v {} {} {}\n".format(vertices[v][0], vertices[v][1], vertices[v][2]))
        for i in range(len(indices)):
            m.writelines("f {} {} {}\n".format(indices[i][0] + 1, indices[i][1] + 1, indices[i][2] + 1))
        m.writelines("\n")
        m.writelines("\n")
        m.close()
    '''
    vertices, indices, mertials = samplesGenerate('', [30, 30, 20])
    vertices = numpy.array(vertices).reshape(-1).tolist()
    indices = numpy.array(indices).reshape(-1).tolist()
    js = {"vertices": vertices, "indices": indices, "layers_of_triangles": mertials, "layer_names": ['box', 'sample']}
    with open("30_30_20.json", 'w') as file_obj:
        json.dump(js, file_obj)
    print("done")



