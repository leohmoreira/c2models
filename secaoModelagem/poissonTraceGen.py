import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import random

def func(x,a,b):

	return a * np.exp(-b*x)

traceSerie = []
trace = []
traceInterval = []
axisX = []
alfa = 0.196
qtdeSimulacoes = 10
for v in range(0,qtdeSimulacoes):
	print 'Simulando ', v, 'de ', qtdeSimulacoes
	trace.append([])
	traceInterval.append([])
	traceSerie.append([])
	axisX = []
	to= 0
	random.seed()
	while to < 30 * 24 * 60:
		to = to + random.expovariate(alfa)
		trace[v].append(to)

	for i in range(0,len(trace[v])-1):
		traceInterval[v].append(trace[v][i+1] - trace[v][i])

	for t in np.arange(0,60,1):        
	        traceSerie[v].append(float(len([q for q in traceInterval[v] if (t < q <= (t+1))])))
	        axisX.append(1 + t)
	
	total = np.sum(traceSerie[v])
	traceSerie[v] = [q/float(total) for q in traceSerie[v]]


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
for t in np.arange(1,60):
	funcao.append(alfa * np.exp(-alfa*t))
	funcaoX.append(t)
plt.close('all')
fig = plt.figure()
plt.plot(funcaoX,funcao,'bo-',
	axisX,traceSerieFinal,'r*-',
	#axisXB,traceSerieB,'g*-'
	)
fig.suptitle("Inter-arrival time")
plt.ylabel("Quantity [Units]")
plt.xlabel("Interval [minutes]")
plt.xticks(axisX,rotation=45)
plt.grid(True)
plt.show()