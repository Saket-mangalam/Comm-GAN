import csv
import matplotlib.pyplot as plt

with open('logbook/test_ber/291870.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    loss0 = []
    for row in csv_reader:
        loss0.append(float(row[1]))
        line_count += 1
print('Processed {line_count} lines.')


with open('logbook/test_ber/714888.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    loss1 = []
    for row in csv_reader:
        loss1.append(float(row[1]))
        line_count += 1
print('Processed {line_count} lines.')

with open('logbook/test_ber/027428.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    loss3 = []
    for row in csv_reader:
        loss3.append(float(row[1]))
        line_count += 1
print('Processed {line_count} lines.')

with open('logbook/test_ber/703606.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    loss5 = []
    for row in csv_reader:
        loss5.append(float(row[1]))
        line_count += 1
print('Processed {line_count} lines.')

#with open('logbook/test_ber/508895.csv') as csv_file:
#    csv_reader = csv.reader(csv_file, delimiter=',')
#    line_count = 0
#    loss8 = []
#        loss8.append(float(row[1]))
#        line_count += 1
#print('Processed {line_count} lines.')

fig = plt.figure()
ax = plt.axes()

x = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
ax.plot(x,loss0)
ax.plot(x,loss1)
ax.plot(x,loss3)
ax.plot(x,loss5)
#ax.plot(x,loss8)
ax.legend(['Noise 0','Noise 0.1','Noise 0.3','Noise 0.5','Noise 0.8'])
ax.set(title="Testing of different trainings", xlabel="SNR", ylabel="BER")

plt.show()