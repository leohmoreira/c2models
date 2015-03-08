# -*- coding: utf-8 -*- 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit,newton
from scipy.stats import lomax, bayes_mvs,expon
import numpy as np
import scipy as sp
import scipy.stats
import math
import random

def funcWeibull(x,a,b):

	#return 1- np.exp(-np.power(x/b,a))
	return 1 - (np.exp(-(np.power((x/a),b))))

def invFuncWeibull(y,a,b):

	return a*np.power((-np.log(1-y)),1.0/b)

def funcExpo(x,a):

	return 1 - np.exp(-a*x)

def intervaloConfianca(limiteTempo,coeficientes,confidence):
	"""
	Calcula o intervalo de confianca a partir da simulação usando a funcao e seus parametros
	A simulação para no tempoLimite = distancia entre o primeiro e último evento
	"""
	
	qtdeSimulacoes = 100
	serieSimulacao = []
	serieIntervaloEntreChegadas = []
	serieQtdeIntervalo = []
	serieProbIntervalo = []
	for simulacao in range(0,qtdeSimulacoes):
		serieSimulacao.append([])
		serieIntervaloEntreChegadas.append([])
		serieQtdeIntervalo.append([])
		serieProbIntervalo.append([])
		tempoSimulacao = 0

		while tempoSimulacao < limiteTempo:
			tempoSimulacao = tempoSimulacao + invFuncWeibull(np.random.uniform(0,1),2.27216869,0.60880361)
			serieSimulacao[simulacao].append(tempoSimulacao)
			"""
			#tempoSimulacao = tempoSimulacao + random.expovariate(coeficientes)
			#descobrindo qual intervalo ----> Monte Carlo
			randomNumber = np.random.uniform(0,1)
			for t in np.arange(0,61,1):
				if(funcWeibull(t,2.27216869,0.60880361) > randomNumber):
					#tempoSimulacao = tempoSimulacao + invFuncWeibull(np.random.uniform(0,1),2.27216869,0.60880361)
					#serieSimulacao[simulacao].append(tempoSimulacao)
					#tempoSimulacao += tempoSimulacao + t
					#serieSimulacao[simulacao].append(t)
					break 
			"""
		# calculo do intervalo entre chegadas
		for i in range(0,len(serieSimulacao[simulacao])-1):
			serieIntervaloEntreChegadas[simulacao].append((serieSimulacao[simulacao][i+1] - serieSimulacao[simulacao][i]))

		#calculo da qtde de chegadas em cada intervalo
		for t in np.arange(0,61,1):        
			serieQtdeIntervalo[simulacao].append(float(len([q for q in serieIntervaloEntreChegadas[simulacao] if (q <= t)])))

		#transformando qtde em porcentagem
		for t in np.arange(0,61,1):        
			serieProbIntervalo[simulacao] = [q/float(serieQtdeIntervalo[simulacao][-1]) for q in serieQtdeIntervalo[simulacao]]
		
	# calculo do IC para cada ponto da distribuicao
	media = []
	upper = []
	lower = []
	valoresPorInstante = []
	for tempo in np.arange(0,61,1):
		valoresPorInstante.append([])
		for simulacao in range(0,qtdeSimulacoes):
			#valoresPorInstante[tempo].append(serieQtdeIntervalo[simulacao][tempo])
			valoresPorInstante[tempo].append(serieProbIntervalo[simulacao][tempo])
		
		# gerando media e limites do IC
		icmedia = str(bayes_mvs(valoresPorInstante[tempo],confidence)).split(')),')[0]
		icmedia = icmedia.replace(" ","")
		icmedia = icmedia.replace("(","")
		icmedia = icmedia.replace(")","")
		m,l,u = icmedia.split(',')

		media.append(m)
		lower.append(l)
		upper.append(u)
		
	return serieProbIntervalo, media,lower,upper

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

if __name__ == "__main__":

	"""
	para gerar a simulação ao inves de simplesmente geram um rando uniform e usar a inversa da CDF, 
	gerer um random uniform e verificar em qual intervalo ele se enquadra (F(x+1) - F(x)) e ai soma este valor em t
	MonteCarlo - Maj David PFC / Maj Camilo
	"""

	plt.close('all')
	fig = plt.figure()
	series, media, lower, upper = intervaloConfianca(1000,0.7,0.99)
	
	#for serie in series:
	#	plt.plot(np.arange(0,61,1),serie,'ro-')		

	print 'Media =', media[1], ' Lower = ', lower[1], ' Upper = ', upper[1]
	plt.plot(np.arange(0,61,1),lower,'bo--')
	plt.plot(np.arange(0,61,1),upper,'bo--')
	plt.plot(np.arange(0,61,1),media,'yo-')

	#serie modelada
	#plt.plot(np.arange(0,61,1),funcExpo(np.arange(0,61,1),0.7),'g*-')
	plt.plot(np.arange(0,61,1),funcWeibull(np.arange(0,61,1),2.27216869,0.60880361),'g*-')

	plt.show()
	