import csv
import matplotlib.pyplot as plt
import glob
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
from matplotlib.cbook import get_sample_data

import numpy as np
import re

with open('1bit9by1,simple/main1.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    ber1 = []
    for row in csv_reader:
        if line_count == 0:
            print('Column names are {", "'.join(row), '}')
            line_count += 1
        elif line_count == 31661:
            break
        else:
            ber1.append(float(row[-1]))
            line_count += 1
print(len(ber1))

with open('1bit9by1,simple/main2.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    ber2 = []
    for row in csv_reader:
        if line_count == 0:
            print('Column names are {", "'.join(row), '}')
            line_count += 1
        elif line_count == 31661:
            break
        else:
            ber2.append(float(row[-1]))
            line_count += 1
print(len(ber2))

with open('1bit9by1,simple/main3.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    ber3 = []
    for row in csv_reader:
        if line_count == 0:
            print('Column names are {", "'.join(row), '}')
            line_count += 1
        elif line_count == 31661:
            break
        else:
            ber3.append(float(row[-1]))
            line_count += 1
print(len(ber3))

with open('1bit9by1,simple/main4.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    ber4 = []
    for row in csv_reader:
        if line_count == 0:
            print('Column names are {", "'.join(row), '}')
            line_count += 1
        elif line_count == 31661:
            break
        else:
            ber4.append(float(row[-1]))
            line_count += 1
print(len(ber4))

fig,ax = plt.subplots()
ax.set_ylim(0,1.0)
for i in range(len(ber1)):
    p1, = ax.plot(i, ber1[i], ".r")
    p2, = ax.plot(i, ber2[i], ".g")
    p3, = ax.plot(i, ber3[i], ".b")
    p4, = ax.plot(i, ber4[i], ".y")
    if i==0:
        # Annotate the 1st position with a text box ('Test 1')
        offsetbox1 = TextArea("Type1", minimumdescent=False)

        ab = AnnotationBbox(offsetbox1, (i,0.4),
                            xybox=(1., -1.),
                            xycoords='data',
                            boxcoords="offset points",
                            arrowprops=dict(arrowstyle="->"))
        ax.add_artist(ab)

        # Annotate the 1st position with a text box ('Test 1')
        offsetbox2 = TextArea("Type2", minimumdescent=False)

        ab = AnnotationBbox(offsetbox2, (i, 0.5),
                            xybox=(1., -1.),
                            xycoords='data',
                            boxcoords="offset points",
                            arrowprops=dict(arrowstyle="->"))
        ax.add_artist(ab)

        # Annotate the 1st position with a text box ('Test 1')
        offsetbox3 = TextArea("Type3", minimumdescent=False)

        ab = AnnotationBbox(offsetbox3, (i, 0.6),
                            xybox=(1., -1.),
                            xycoords='data',
                            boxcoords="offset points",
                            arrowprops=dict(arrowstyle="->"))
        ax.add_artist(ab)

        # Annotate the 1st position with a text box ('Test 1')
        offsetbox4 = TextArea("Type4", minimumdescent=False)

        ab = AnnotationBbox(offsetbox4, (i, 0.7),
                            xybox=(1., -1.),
                            xycoords='data',
                            boxcoords="offset points",
                            arrowprops=dict(arrowstyle="->"))
        ax.add_artist(ab)

    if i%5000 == 0 and i != 0:
        fn = "1bit9by1,simple/main1/fake_enc%d.png"%i
        arr_img = plt.imread(fn, format='png')
        arr_img = arr_img[136:201,136:201]

        imagebox1 = OffsetImage(arr_img, zoom=0.5)
        imagebox1.image.axes = ax

        ab1 = AnnotationBbox(imagebox1, (i,0.4),
                            xybox=(1.,-1.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.0,
                            arrowprops=dict(
                                arrowstyle="->")
                            )

        ax.add_artist(ab1)

        fn = "1bit9by1,simple/main2/fake_enc%d.png" % i
        arr_img = plt.imread(fn, format='png')
        arr_img = arr_img[136:201, 136:201]

        imagebox2 = OffsetImage(arr_img, zoom=0.5)
        imagebox2.image.axes = ax

        ab2 = AnnotationBbox(imagebox2, (i, 0.5),
                             xybox=(1., -1.),
                             xycoords='data',
                             boxcoords="offset points",
                             pad=0.0,
                             arrowprops=dict(
                                 arrowstyle="->")
                             )

        ax.add_artist(ab2)

        fn = "1bit9by1,simple/main3/fake_enc%d.png" % i
        arr_img = plt.imread(fn, format='png')
        arr_img = arr_img[136:201, 136:201]

        imagebox3 = OffsetImage(arr_img, zoom=0.5)
        imagebox3.image.axes = ax

        ab3 = AnnotationBbox(imagebox3, (i, 0.6),
                             xybox=(1., -1.),
                             xycoords='data',
                             boxcoords="offset points",
                             pad=0.0,
                             arrowprops=dict(
                                 arrowstyle="->")
                             )

        ax.add_artist(ab3)

        fn = "1bit9by1,simple/main4/fake_enc%d.png" % i
        arr_img = plt.imread(fn, format='png')
        arr_img = arr_img[136:201, 136:201]

        imagebox4 = OffsetImage(arr_img, zoom=0.5)
        imagebox4.image.axes = ax

        ab4 = AnnotationBbox(imagebox4, (i, 0.7),
                             xybox=(1., -1.),
                             xycoords='data',
                             boxcoords="offset points",
                             pad=0.0,
                             arrowprops=dict(
                                 arrowstyle="->")
                             )

        ax.add_artist(ab4)

plt.xlabel("BER")
plt.ylabel("Batch number")
plt.legend((p1, p2, p3, p4), ('Type1', 'Type2', 'Type3', 'Type4'))
plt.show()
plt.savefig('graph.png')