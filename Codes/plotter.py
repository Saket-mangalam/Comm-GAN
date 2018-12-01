# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 08:29:55 2018

@author: saket
"""

import csv
import matplotlib.pyplot as plt
#import numpy as np
import re

with open('logbook/262468.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    g = []
    d= [] 
    dec = []
    ber=[]
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            g.append(float(row[1]))
            d.append(float(row[2]))
            dec.append(float(row[3]))
            ber.append(float(re.findall(r"[-+]?\d*\.\d+|\d+", row[4])[0]))
            line_count += 1
print(f'Processed {line_count} lines.')

fig = plt.figure()
ax = plt.axes()

x = list(range(len(g)))
ax.plot(x,g)
ax.plot(x,d)
ax.plot(x,dec)
ax.plot(x,ber)
ax.legend(['gloss','dloss','decloss','ber'])
ax.set(title="262468", xlabel="batches done", ylabel="losses")
