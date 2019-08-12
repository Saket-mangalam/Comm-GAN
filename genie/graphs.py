import csv
import matplotlib.pyplot as plt
import glob
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
from matplotlib.cbook import get_sample_data

import numpy as np
import re

with open('images/673049/673049.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    nber, nlber = [], []
    for row in csv_reader:
        if line_count == 0:
            print('Column names are {", "'.join(row), '}')
            line_count += 1
        else:
            nber.append(float(row[7]))
            nlber.append(float(row[8]))
            line_count += 1
print(len(nber))
x = np.linspace(1, len(nber), len(nber))
fig,ax = plt.subplots()
ax.set_ylim(0,2.0)
p1, = ax.plot(x, nber, "r")
p2, = ax.plot(x, nlber, "g")
"""
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
    """
plt.ylabel("MSE LOSS")
plt.xlabel("Batch number")
plt.legend((p1, p2), ('Noiseless', 'Noisy'))
plt.show()
plt.savefig('images/00.png',fig)