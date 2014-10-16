import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
def func(x,a,b):

	#return a * b**x
	#x = np.array(x)
	
	return a * np.exp(-b*x)

to = 0
traceSerie = []
trace = []
traceInterval = []
axisX = []
while to < 20 * 24 * 3600:
	to = to + 60
	if np.random.uniform(0,1) <= 0.5:
		trace.append(to)

for i in range(0,len(trace)-1):
	traceInterval.append(trace[i+1] - trace[i])

for i in np.arange(0,2*3600,300):
	traceSerie.append(len([t for t in traceInterval if t>i and t<=i+300]))
	axisX.append(i)
print len(traceSerie)	
print "---- = ", traceSerie

popt, pocv = curve_fit(func,axisX,traceSerie)
print popt
index = np.arange(len(axisX))
plt.bar(axisX,traceSerie,0.3)
plt.show()