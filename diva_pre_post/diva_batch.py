# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:46:02 2018

@author: B
"""

import os
import subprocess
import numpy as np

imageGroundTruth=[]
xmlGroundTruth=[]
xmlPrediction=[]
overlap=[]
outputPath='../overlap_output/'
csv=False


imageGroundTruth_folder='CB55_image_gt/'
xmlGroundTruth_folder='CB55_xml_gt/'
xmlPrediction_folder='cb55_xml_prediction/'
original_image_folder='CB55/'

imageGroundTruth=os.listdir(imageGroundTruth_folder)
xmlGroundTruth=os.listdir(xmlGroundTruth_folder)
xmlPrediction=os.listdir(xmlPrediction_folder)
original_image=os.listdir(original_image_folder)
number_of_files=len(os.listdir(imageGroundTruth_folder))
args=[]
for i in range(number_of_files):
    args.append('-igt ')
    args.append(imageGroundTruth_folder + imageGroundTruth[i])
    args.append(' -xgt ')
    args.append(xmlGroundTruth_folder + xmlGroundTruth[i])
    args.append(' -xp ')
    args.append(xmlPrediction_folder+xmlPrediction[i])
    args.append(' -overlap ')
    args.append(original_image_folder + original_image[i])
    args.append(' -out ')
    args.append(outputPath)

lineIU=[]
lineF1=[]
f=open(original_image_folder[:-1]+'_results.txt','w')

for i in range(number_of_files):
    p = subprocess.Popen(['java', '-jar',
                          'C:/Users/berat/Downloads/DIVA_Line_Segmentation_Evaluator-master/out/artifacts/LineSegmentationEvaluator.jar',
                          args[i*10:i*10+10]],
                         cwd='C:/Users/berat/PycharmProjects/diva_pre_post/', universal_newlines=True,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,bufsize=1)
    stdO,stdE = p.communicate()

    ind=stdO.find('line IU')
    lineiu=stdO[ind+10:ind+15]
    if '1.0' in lineiu:
        lineiu = float(stdO[ind + 10:ind + 13])
    else:
        lineiu = float(stdO[ind + 10:ind + 15])

    lineIU.append(lineiu)

    ind = stdO.find('line F1')
    linef1 = stdO[ind + 10:ind + 15]
    if '1.0' in linef1:
        linef1 = float(stdO[ind + 10:ind + 13])
    else:
        linef1 = float(stdO[ind + 10:ind + 15])

    lineF1.append(linef1)

    print('current lineiu ', lineIU)
    print('current linef1 ', lineF1)
    filename=args[i*10+1].split('/')[1]
    print(filename)
    f.write(filename+'\t line IU=' +str(lineiu)+'\t line F1=' +str(linef1)+'\n')

mean_lineiu=np.mean(lineIU)
mean_linef1=np.mean(lineF1)

print('mean line iu = ', mean_lineiu)
print('mean line f1 = ', mean_linef1)
f.write('\n\n\n')
f.write('mean line iu = '+ str(mean_lineiu)+'\n')
f.write('mean line f1 = '+ str(mean_linef1))

f.close()