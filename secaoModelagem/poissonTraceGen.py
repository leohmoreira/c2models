import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit,newton
from scipy.stats import lomax, bayes_mvs
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
qtdeSimulacoes = 100
a= 0.43045898


b=-2.673e-4
plt.close('all')
fig = plt.figure()
for v in range(0,qtdeSimulacoes):
	print 'Simulando ', v, 'de ', qtdeSimulacoes
	trace.append([])
	traceInterval.append([])
	traceSerie.append([])
	axisX = []
	to= 0
	random.seed()
	
	while to < 30 * 24 * 60:
		#to = to -np.log(1-np.random.uniform(0.0,1.0))/a
		to = to + np.random.pareto(a)
		#to = to + np.random.exponential(1/a)

		#print to
		trace[v].append(to)
	
	#traceInterval[v] = np.random.pareto(a,1000)
	#traceInterval[v] = np.random.exponential(1/a,1000)
	
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
lower=[]
upper=[]
media=[]
posicao=[]
for x in range(0,len(traceSerie[0])):
	posicao=[]
	valor=0
	for q in range(0,qtdeSimulacoes):
		valor = valor + traceSerie[q][x]
		posicao.append(traceSerie[q][x])
	valor = valor/float(qtdeSimulacoes)
	icmedia = str(bayes_mvs(posicao,0.99)).split(')),')[0]
	icmedia = icmedia.replace(" ","")
	icmedia = icmedia.replace("(","")
	icmedia = icmedia.replace(")","")
	m,l,u = icmedia.split(',')
	

	media.append(float(m))
	lower.append(float(l))
	upper.append(float(u))
	
	traceSerieFinal.append(valor)


funcao=[]
"""
for t in np.arange(0,30,1):
	funcao.append(a/float(np.power(t+1,a+1)))
	#funcao.append(a*np.exp(-a*t))
	axisX.append(t)
axisX = axisX
"""
plt.plot(
	#axisX,funcao,'bo-',
	axisX,lower,'ro--',
	axisX,upper,'go--',
	axisX,media,'yo-',
	)
fig.suptitle("Inter-arrival time")
plt.ylabel("Quantity [Units]")
plt.xlabel("Interval [minutes]")
plt.xticks(axisX,rotation=45)
plt.grid(True)
plt.show()