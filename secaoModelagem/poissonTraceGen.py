import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit,newton
from scipy.stats import lomax, bayes_mvs,expon
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
qtdeSimulacoes = 10
a= 0.37608875


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
		to = to + random.expovariate(a)
		#to = to + np.random.pareto(a)
		#to = to + np.random.exponential(1/a)

		#print to
		trace[v].append(to)
	
	#traceInterval[v] = np.random.pareto(a,500)
	#traceInterval[v] = np.random.exponential(1/a,500)
	
	
	for i in range(0,len(trace[v])-1):
		traceInterval[v].append(trace[v][i+1] - trace[v][i])
	
	for t in np.arange(0,61,1):        
	        traceSerie[v].append(float(len([q for q in traceInterval[v] if (q <= t)])))
	        #traceSerie[v].append(float(len([q for q in traceInterval[v] if (t < q <= (t+1))])))
	        axisX.append(t)
		
	#total = np.sum(traceSerie[v])
	total = float(len(trace[v])-1)
	total = (traceSerie[v][-1])
	#print total,np.sum(traceSerie[v]),traceSerie[v]
	if(total > 0):
		traceSerie[v] = [float(q)/float(total) for q in traceSerie[v]]

		plt.plot(axisX,traceSerie[v],'yo-')
	
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
	traceSerieFinal.append(valor)
	"""
	icmedia = str(bayes_mvs(posicao,0.99)).split(')),')[0]
	icmedia = icmedia.replace(" ","")
	icmedia = icmedia.replace("(","")
	icmedia = icmedia.replace(")","")
	m,l,u = icmedia.split(',')
	

	media.append(float(m))
	lower.append(float(l))
	upper.append(float(u))
	"""
	


funcao=[]
funcaoB=[]

for t in np.arange(0,61,1):
	#funcao.append(a/float(np.power(t+1,a+1)))
	#funcaoB.append(lomax.pdf(t,a))
	funcao.append(1.0 - np.exp(-a*t))
	#funcaoB.append(expon.pdf(t))
	
plt.plot(
	axisX,funcao,'bo-',
	#axisX,funcaoB,'ro-',
	#axisX,lower,'ro--',
	#axisX,upper,'go--',
#	axisX,traceSerieFinal,'yo-',
	)
fig.suptitle("Inter-arrival time")
plt.ylabel("Quantity [Units]")
plt.xlabel("Interval [minutes]")
plt.xticks(axisX,rotation=45)
plt.grid(True)
plt.show()