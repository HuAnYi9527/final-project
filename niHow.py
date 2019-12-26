import matplotlib.pyplot as plt
import numpy as np
import time
import time, random
import math
import serial
from collections import deque
from scipy.io import loadmat
from scipy import signal
import statistics

#Display loading 
class PlotData:
    def __init__(self, max_entries=30):
        self.axis_x = deque(maxlen=max_entries)
        self.axis_y = deque(maxlen=max_entries)
        self.xff = deque(maxlen= 100)

    def add(self, x, y):
        self.axis_x.append(x)
        self.axis_y.append(y)
    def xf(self, fs):
        self.xff = np.fft.fft(self.axis_y)
        self.fs = (np.arange(-np.pi, np.pi, np.pi*2/(len(self.axis_y))))*fs/(np.pi)

    def filter(self, kernel): 
        self.filtered = signal.lfilter(kernel, 1, self.axis_y)

#initial
HRV = deque(maxlen= 200)
kernel = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6] 
angle = np.linspace(-np.pi, np.pi, 50)
cirx = np.sin(angle)
ciry = np.cos(angle)

z = np.roots(kernel)

plt.figure(figsize=(8,8))
plt.plot(cirx, ciry,'k-')
plt.plot(np.real(z), np.imag(z), 'o',color = 'b', markersize=12)
plt.plot(0, 0, 'x', markersize=12)
plt.plot(1,0,'o',color = 'b',markersize=12)
plt.grid()

plt.xlim((-2, 2))
plt.xlabel('Real')
plt.ylim((-2, 2))
plt.ylabel('Imag')
fig, ((timely_ax,spectrum_ax),(filtered_ax, frequency_response_ax)) = plt.subplots(2,2)
timely_line,  = timely_ax.plot(np.random.randn(100))
spectrum_line, = spectrum_ax.plot(np.random.randn(100))
filtered_line, = filtered_ax.plot(np.random.randn(100))
frequency_response_line, = frequency_response_ax.plot(np.random.randn(100))


plt.show(block = False)
plt.setp(spectrum_line,color = 'r')
plt.setp(filtered_line,color = 'g')
plt.setp(filtered_line,color = 'b')


PData= PlotData(500)
PData2 = PlotData(500)
timely_ax.set_ylim(-10,10)
spectrum_ax.set_ylim(0,500)
filtered_ax.set_ylim(-10,10)
frequency_response_ax.set_ylim(0,1)


# plot parameters
print ('plotting data...')
# open serial port
strPort='com3'
ser = serial.Serial(strPort, 115200)
ser.flush()

start = time.time()
temp = deque(maxlen=10)


#ploting
while True:
    for ii in range(10):
        try:
            data = float(ser.readline())
            temp.append(data)
            PData.add(time.time() - start, data - np.mean(temp))
        except:
            pass



    PData.filter(kernel)
    w, h = signal.freqz(kernel)
    fs = 100
    PData.xf(fs)
    frequency_response = np.fft.fft(PData.filtered)/PData.xff

    list_tmp = np.zeros(len(PData.axis_y))
    for i in range(len(PData.axis_y)):
    	    list_tmp[i] = PData.axis_y[i]

    index_extr,  = signal.argrelextrema(list_tmp, np.greater, order = 40)
    related_extrmax = list_tmp[index_extr]
    try:
        dd = index_extr[4] - index_extr[3]
        HRV.append((60/(dd*0.01)))
        print('即時心率 %f (次／min)' %(60/(dd*0.01)))
        HRV.append((60/(dd*0.01)))
    except:
        pass


    timely_ax.set_xlim(PData.axis_x[0], PData.axis_x[0]+5)
    spectrum_ax.set_xlim(0 , fs)
    filtered_ax.set_xlim(PData.axis_x[0], PData.axis_x[0]+5)
    frequency_response_ax.set_xlim(0, fs)

    timely_line.set_xdata(PData.axis_x)
    timely_line.set_ydata(PData.axis_y)
    spectrum_line.set_xdata(PData.fs)
    spectrum_line.set_ydata(abs(PData.xff))
    filtered_line.set_xdata(PData.axis_x)
    filtered_line.set_ydata(PData.filtered)
    frequency_response_line.set_xdata(PData.fs)
    frequency_response_line.set_ydata(frequency_response)

    fig.canvas.draw()
    fig.canvas.flush_events()
