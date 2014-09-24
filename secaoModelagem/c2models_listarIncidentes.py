# -*- coding: utf-8 -*- 

from incidentes.models import *
from datetime import datetime
import collections
import argparse
import numpy as np
from scipy import special, optimize
import matplotlib.pyplot as plt
import time

def gravar_arquivo(nome_arquivo, modo, info):

	file = open(nome_arquivo,modo)
	file.write(info)


def listar_incidentes():
	
	#obtendo todos os incidentes do banco Incident_handling_copa_conf
	incidentes = Incident.get_all()

	datas_incidentes = [incidente.reporting_date for incidente in incidentes]

	#ordena as datas
	datas_incidentes.sort()

	#recupera no formato ANO/MES/DIA
	porDia_incidentes = [dia.strftime('%Y/%B/%d %H:%m:%S') for dia in datas_incidentes]

	#conta a frequencia de incidentes por dia
	cluster_porDia = collections.Counter(porDia_incidentes)

	resultados={}
	eixoX = []
	eixoY = []

	#f = open("clusterizado_completo.c2", "w")
	for data in sorted(cluster_porDia):
		resultados[data] = cluster_porDia[data]
		#eixoX.append(data)
		eixoY.append(cluster_porDia[data])
		print data, " : ",cluster_porDia[data] 
	#	f.write(str(data)+ " : " + str(cluster_porDia[data])+"\n")
		timeTuple = time.strptime(data, "%Y/%B/%d %H:%m:%S")
		timeStamp = time.mktime(timeTuple)
		eixoX.append(repr(timeStamp))

	#print eixoX

	#plt.plot(eixoX,eixoY)
	#plt.savefig("plot.png",dpi=96)


listar_incidentes()