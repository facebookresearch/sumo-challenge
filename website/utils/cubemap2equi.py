from PIL import Image
import numpy as np
import imageio

# -*- coding: utf-8 -*-

# revitToEquirectangular.py
#
# Takes a panoramic image exported from Revit and converts it into an equirectangular image for use in virtual reality applications
# Works with both panoramic images and combined stereo panoramic images
#
# For use with other cube maps this script will require further editing, for now remember that the script assumes the cube maps were originally in the following format:
#
#   #
#      ####
#       #


import math     # Math functions
import sys      # allows us to extract the function args
import pandas as pd

def get_colors():
    """ Load colors from CSV file and return it as a dictionary of colors
    """
    color_array = pd.read_csv('data/40colors.csv', delimiter=',').values
    colors = {}
    for i in range(color_array.shape[0]):
        colors[i] = color_array[i, :].tolist()
    return colors
def convertStrip(out):

    print('Converting:')
    increment = (dims*2)/100
    counter = 0
    percentCounter = 0

    for j in range(0,int(dims*2)):

        if(counter<=j):
            print(str(percentCounter)+'%')
            percentCounter+=1
            counter+=increment

        v = 1.0 - ((float(j)) / (dims*2))
        phi = v*math.pi

        for i in range(0,int(dims*4)):

            u = (float(i))/(dims*4)
            theta = u * 2 * math.pi

            # all of these range between 0 and 1
            x = math.cos(theta)*math.sin(phi)
            y = math.sin(theta)*math.sin(phi)
            z = math.cos(phi)

            a = max(max(abs(x),abs(y)),abs(z));

            # one of these will equal either -1 or +1
            xx = x / a;
            yy = y / a;
            zz = z / a;

            # format is left, front, right, back, bottom, top;
            # therefore left, front, right, back, bottom, top
            if(yy == -1):                   # square 1 left
                xPixel = int(((-1*math.tan(math.atan(x/y))+1.0)/2.0)*dims)
                yTemp = int(((-1*math.tan(math.atan(z/y))+1.0)/2.0)*(dims-1))
                imageSelect = 1
            elif(xx == 1):          # square 2; front
                xPixel = int(((math.tan(math.atan(y/x))+1.0)/2.0)*dims)
                yTemp = int(((math.tan(math.atan(z/x))+1.0)/2.0)*dims)
                imageSelect = 2
            elif(yy == 1):          # square 3; right
                xPixel = int(((-1*math.tan(math.atan(x/y))+1.0)/2.0)*dims)
                yTemp = int(((math.tan(math.atan(z/y))+1.0)/2.0)*(dims-1))
                imageSelect = 3
            elif(xx == -1):         # square 4; back
                xPixel = int(((math.tan(math.atan(y/x))+1.0)/2.0)*dims)
                yTemp = int(((-1*math.tan(math.atan(z/x))+1.0)/2.0)*(dims-1))
                imageSelect = 4
            elif(zz == 1):          # square 5; bottom
                xPixel = int(((math.tan(math.atan(y/z))+1.0)/2.0)*dims)
                yTemp = int(((-1*math.tan(math.atan(x/z))+1.0)/2.0)*(dims-1))
                imageSelect = 5
            elif(zz == -1):         # square 6; top
                xPixel = int(((-1*math.tan(math.atan(y/z))+1.0)/2.0)*dims)
                yTemp = int(((-1*math.tan(math.atan(x/z))+1.0)/2.0)*(dims-1))
                imageSelect = 6
            else:
                print('error, program should never reach this point')
                sys.exit(0)

            yPixel = (dims-1) if (yTemp>dims-1) else yTemp

            if(yPixel>dims-1):
                yPixel=dims-1
            if(xPixel>dims-1):
                xPixel=dims-1

            if(imageSelect==1):
                output.append(left[int(xPixel),int(yPixel)])
            elif(imageSelect==2):
                output.append(front[int(xPixel),int(yPixel)])
            elif(imageSelect==3):
                output.append(right[int(xPixel),int(yPixel)])
            elif(imageSelect==4):
                output.append(back[int(xPixel),int(yPixel)])
            elif(imageSelect==5):
                output.append(bottom[int(xPixel),int(yPixel)])
            elif(imageSelect==6):
                output.append(top[int(xPixel),int(yPixel)])
            else:
                print('error, program should never reach this point')
                sys.exit(0)


# begin main program code

im = imageio.mimread('data/sumo-input.tif')
labels = ["RGB", "Depth", "Category", "Instance"]
colors = get_colors()

for i in range(len(labels)):
    img = im[i]
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    faces = [img[:, k*1024:(k+1)*1024, :].transpose((1, 0, 2)).squeeze() for k in range(6)]
    back, left, front, right, top, bottom = faces

    dims = left.shape[0]

    xPixel = 0
    yPixel = 0
    yTemp = 0
    imageSelect = 0
    outputHeight = 0
    output = []
    outputWidth = dims*4
    outputHeight = dims*2
    fname = "../static/images/%s_equi.png" % (labels[i])
    convertStrip(output)
    if labels[i] == "Depth":
        img = Image.new("L", ((int(outputWidth)),(int(outputHeight))), None)
    else:
        img = Image.new("RGB", ((int(outputWidth)),(int(outputHeight))), None)

    if type(output[0]) is np.uint16:
        if labels[i] == "Depth":
            output = [(255*x/(2**16)) for x in output]
        else:
            result = []
            for x in output:
                clr = [int(255*kx) for kx in list(colors[x % 39])]
                result.append(tuple(clr))
            output = result
    else:
        output = [tuple(x) for x in output]
    img.putdata(output)
    img.save(fname) # output file name
    #img.show()



