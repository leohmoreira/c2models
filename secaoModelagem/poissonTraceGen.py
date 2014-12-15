import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit,newton
from scipy.stats import lomax
import math
import random

def func(x,a,b):
	return a / float(np.power(x+1,a+1))

traceSerie = []
trace = []
traceInterval = []
axisX = []
#alfa = 0.37608875
alfa = 0.703
qtdeSimulacoes = 3
a=0.430
a= 0.43045898
#a=0.394
b=-2.673e-4
for v in range(0,qtdeSimulacoes):
	print 'Simulando ', v, 'de ', qtdeSimulacoes
	trace.append([])
	traceInterval.append([])
	traceSerie.append([])
	axisX = []
	to= 0
	random.seed()

	"""
	while to < 30 * 24 * 60:
		#to = to + lomax.ppf(np.random.uniform(0.0,1.0),a)
		to = to + np.random.pareto(a)
		print to
		trace[v].append(to)
	"""
	traceInterval[v] = np.random.pareto(a,30*24*60)
	#traceInterval[v] = np.random.exponential(a,30*24*60)
	
	for i in range(0,len(trace[v])-1):
		traceInterval[v].append(trace[v][i+1] - trace[v][i])

	for t in np.arange(0,30,1):        
	        traceSerie[v].append(float(len([q for q in traceInterval[v] if (t <= q < (t+1))])))
	        axisX.append(t)
	
	
	total = np.sum(traceSerie[v])
	if(total > 0):
		traceSerie[v] = [q/float(total) for q in traceSerie[v]]
	else:
		v = v -1


traceSerieFinal=[]
for x in range(0,len(traceSerie[0])):
	valor=0
	for q in range(0,qtdeSimulacoes):
		valor = valor + traceSerie[q][x]
	valor = valor/float(qtdeSimulacoes)
	traceSerieFinal.append(valor)
print len(axisX), len(traceSerieFinal)
funcao=[]
funcaoX=[]
for t in np.arange(0,30,1):
	funcao.append(a/float(np.power(t+1,a+1)))
	funcaoX.append(t)
plt.close('all')
fig = plt.figure()
plt.plot(
	funcaoX,funcao,'bo-',
	axisX,traceSerieFinal,'r*-',
	#axisXB,traceSerieB,'g*-'
	)
fig.suptitle("Inter-arrival time")
plt.ylabel("Quantity [Units]")
plt.xlabel("Interval [minutes]")
plt.xticks(axisX,rotation=45)
plt.grid(True)
plt.show()