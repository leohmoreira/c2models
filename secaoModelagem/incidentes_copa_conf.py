# -*- coding: utf-8 -*- 	
import datetime
import collections
import argparse
import numpy as np
from scipy import special, optimize
import matplotlib.pyplot as plt
import time
from matplotlib.dates import DateFormatter
import os, sys
lib_path_Pacificador = os.path.abspath('/home/moreira/Projetos/COP/pacificador_cop')
sys.path.append(lib_path_Pacificador)
from incidentes.models import *

def read_file(fileName):
    '''
	Esta funcao lê o arquivo e retorna um list com DATA e QTDE
    '''    
    return tuple(open(fileName, 'r'))
    
def date_converter(data):
	
	return data[:data.index(":")-1]

def qtd_incidents(data):
	
	return data[data.index(":")+2:-1]

def plot_graph(axisX, axisY):

    fig, ax = plt.subplots()
    ax.set_title("Incidentes na Copa das Confederacoes - Jun 13")
    ax.set_ylabel("Quantidade")
    ax.set_xlabel("Dias")
    ax.plot_date(axisX,axisY,fmt='-ro',xdate=True,ydate=False)
    plt.xticks(axisX,rotation=90)
    ax.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.grid(True)
    plt.show()
    plt.savefig("plot.png",dpi=96)

def get_Incidents_from_DB(inicioCopaConf,terminoCopaConf):

    clusteringDefault = '%Y/%m/%d'
    #clusterPorHora = '%Y/%B/%d %H:%m:%S'
    incidents = Incident.get_all()
    incidentsDates = [
                        (i.reporting_date).strftime(clusteringDefault) for i in incidents 
                        if (inicioCopaConf <= i.reporting_date) and (i.reporting_date <=terminoCopaConf)
                    ]
    
    return incidentsDates

def get_Incidentes_from_file(inicioCopaConf, terminoCopaConf):

    datesFromFile = read_file('clusterizado.c2')
    dates = map(date_converter, datesFromFile)
    incidentes = map(qtd_incidents, datesFromFile)
    datesIncidents = {}
    for i, v in enumerate(dates):
        tmpDate = datetime.strptime(v,"%Y/%B/%d")
        if (inicioCopaConf <= tmpDate) and (tmpDate <= terminoCopaConf):

            datesIncidents[time.strftime("%Y/%m/%d",time.strptime(v,"%Y/%B/%d"))] = incidentes[i]

    return datesIncidents

if __name__ == "__main__":
    
    inicioCopaConf = datetime(2013,6,10)
    terminoCopaConf = datetime(2013,7,3)
    
    # para recuperar do banco
    
    incidentsDates =  get_Incidents_from_DB(inicioCopaConf,terminoCopaConf)
    dictionaryIncidentsDates = {}
    #Faz a contagem de incidentes por data segundo o padrão de cluster
    clusterIncidentes = collections.Counter(incidentsDates)

    for i,v in enumerate(incidentsDates):
        dictionaryIncidentsDates[v]=clusterIncidentes[v]
    
    """   
    #para recuperar de um arquivo já clusterizado
    dictionaryIncidentsDates = get_Incidentes_from_file(inicioCopaConf,terminoCopaConf)
    """
    sortedIncidentDates = sorted(set(dictionaryIncidentsDates))

    axisX = []
    axisY = []
    for key in sortedIncidentDates:
        axisX.append(datetime.strptime(key,"%Y/%m/%d"))
        axisY.append(dictionaryIncidentsDates[key])
    
    plot_graph(axisX,axisY)
    