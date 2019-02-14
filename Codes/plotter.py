# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 08:29:55 2018

@author: saket
"""

import csv
import matplotlib.pyplot as plt
#import numpy as np
#import re

with open('logbook/430465.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    g1 = []
    d1 = []
    dec1 = []
    ber1 = []
    for row in csv_reader:
        if line_count == 0:
            print('Column names are {", ".join(row)}')
            line_count += 1
        else:
            g1.append(float(row[2])+float(row[4]))
            d1.append(float(row[1]))
            dec1.append(float(row[3]))
            ber1.append(float(row[5]))
            line_count += 1
print('Processed {line_count} lines.')

with open('logbook/106568.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    g2 = []
    d2 = []
    dec2 = []
    ber2 = []
    for row in csv_reader:
        if line_count == 0:
            print('Column names are {", ".join(row)}')
            line_count += 1
        else:
            g2.append(float(row[2])+float(row[4]))
            d2.append(float(row[1]))
            dec2.append(float(row[3]))
            ber2.append(float(row[5]))
            line_count += 1
print('Processed {line_count} lines.')

with open('logbook/567447.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    g3 = []
    d3 = []
    dec3 = []
    ber3 = []
    for row in csv_reader:
        if line_count == 0:
            print('Column names are {", ".join(row)}')
            line_count += 1
        else:
            g3.append(float(row[2])+float(row[4]))
            d3.append(float(row[1]))
            dec3.append(float(row[3]))
            ber3.append(float(row[5]))
            line_count += 1
print('Processed {line_count} lines.')

with open('logbook/500595.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    g4 = []
    d4 = []
    dec4 = []
    ber4 = []
    for row in csv_reader:
        if line_count == 0:
            print('Column names are {", ".join(row)}')
            line_count += 1
        else:
            g4.append(float(row[2])+float(row[4]))
            d4.append(float(row[1]))
            dec4.append(float(row[3]))
            ber4.append(float(row[5]))
            line_count += 1
print('Processed {line_count} lines.')

with open('logbook/854309.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    g5 = []
    d5 = []
    dec5 = []
    ber5 = []
    for row in csv_reader:
        if line_count == 0:
            print('Column names are {", ".join(row)}')
            line_count += 1
        else:
            g5.append(float(row[2])+float(row[4]))
            d5.append(float(row[1]))
            dec5.append(float(row[3]))
            ber5.append(float(row[5]))
            line_count += 1
print('Processed {line_count} lines.')

fig = plt.figure()
ax = plt.axes()

x = list(range(len(g1)))
ax.plot(x,ber1)
#ax.plot(x,ber2)
#ax.plot(x,ber3)
#ax.plot(x,ber4)
#ax.plot(x,ber5)
#ax.legend(['Noise 0','Noise 0.1','Noise 0.3','Noise 0.5','Noise 0.8'])
ax.set(title="Disc:dec 1:999", xlabel="batches done", ylabel="losses")

plt.show()