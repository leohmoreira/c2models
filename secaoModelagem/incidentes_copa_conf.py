# -*- coding: utf-8 -*- 	
import datetime
import collections
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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

def plot_graph_date(axisX, axisY):

    matchDays = [datetime(2013,6,10), datetime(2013,6,30)]
    fig, ax = plt.subplots()
    ax.set_title("Incidentes na Copa das Confederacoes - Jun 13")
    ax.set_ylabel("Quantidade")
    ax.set_xlabel("Dias")
    ax.plot_date(axisX,axisY,fmt='-ro',xdate=True,ydate=False)
    plt.xticks(axisX,rotation=90)
    ax.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.grid(True)
    plt.show()
    #plt.savefig("plot.png",dpi=96)

def plot_graph_actions(axisX, axisY):

    fig, graph1 = plt.subplots()
    graph1.set_title("Incidentes na Copa das Confederacoes - Jun 13")
    graph1.set_ylabel("Quantidade")
    graph1.set_xlabel("Dias")
    graph1.plot(axisX,axisY, 'bo')
    graph1.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graph1.grid(True)
    plt.show()
    #plt.savefig("plot.png",dpi=96)

def plot_graph(axisX, axisY):

    fig, graph1 = plt.subplots()
    graph1.set_title("Incidentes na Copa das Confederacoes - Jun 13")
    graph1.set_ylabel("Quantidade")
    graph1.set_xlabel("Dias")
    graph1.plot(axisX,axisY, 'bo')
    graph1.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graph1.grid(True)
    plt.show()
    #plt.savefig("plot.png",dpi=96)

def plot_total(graphType,
                axisXtotal, axisYtotal,
                axisX1, axisY1,
                axisX2, axisY2,
                axisX3, axisY3,
                axisX4, axisY4,
                axisX5, axisY5,
                axisX6, axisY6,
                axisX7, axisY7,
                axisX8, axisY8,
            ):

    plt.close('all')
    fig = plt.figure()

    graphTotal = plt.subplot2grid((4,3),(0,0),colspan=3)
    graphCOC = plt.subplot2grid((4,3),(1,0),)
    graphCC2FTCSSA = plt.subplot2grid((4,3),(1,2))
    graphFOR = plt.subplot2grid((4,3),(2,0))
    graphREC = plt.subplot2grid((4,3),(2,1))
    graphSSA = plt.subplot2grid((4,3),(2,2))
    graphRIO = plt.subplot2grid((4,3),(3,0))
    graphBSB = plt.subplot2grid((4,3),(3,1))
    graphBHZ = plt.subplot2grid((4,3),(3,2))

    graphTotal.set_title("Total de incidentes na Copa das Confederacoes - Jun 13")
    graphTotal.set_ylabel("Quantidade")
    graphTotal.set_xlabel("Dias")
    if (graphType == 'line'):
        graphTotal.plot(axisX,axisY, 'bo-')
    elif(graphType == 'bar'):
        graphTotal.bar(axisX,axisY, align='center')
        graphTotal.xaxis_date()

    graphTotal.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphTotal.grid(True)

    graphCOC.set_title("COC")
    graphCOC.set_ylabel("Quantidade")
    graphCOC.set_xlabel("Dias")
    if (graphType == 'line'):
        graphCOC.plot(axisX1,axisY1, 'bo-')
    elif(graphType == 'bar'):
        graphCOC.bar(axisX1,axisY1, align='center')
        graphCOC.xaxis_date()

    graphCOC.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX1,rotation=90)
    graphCOC.grid(True)

    graphCC2FTCSSA.set_title("CC2 - FTC - SSA")
    graphCC2FTCSSA.set_ylabel("Quantidade")
    graphCC2FTCSSA.set_xlabel("Dias")
    if (graphType == 'line'):
        graphCC2FTCSSA.plot(axisX2,axisY2, 'bo-')
    elif(graphType == 'bar'):
        graphCC2FTCSSA.bar(axisX2,axisY2, align='center')
        graphCC2FTCSSA.xaxis_date()
    graphCC2FTCSSA.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX2,rotation=90)
    graphCC2FTCSSA.grid(True)

    graphFOR.set_title("FORTALEZA")
    graphFOR.set_ylabel("Quantidade")
    graphFOR.set_xlabel("Dias")
    if (graphType == 'line'):
        graphFOR.plot(axisX3,axisY3, 'bo-')
    elif(graphType == 'bar'):
        graphFOR.bar(axisX3,axisY3, align='center')
        graphFOR.xaxis_date()
    graphFOR.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX3,rotation=90)
    graphFOR.grid(True)

    graphREC.set_title("RECIFE")
    graphREC.set_ylabel("Quantidade")
    graphREC.set_xlabel("Dias")
    if (graphType == 'line'):
        graphREC.plot(axisX4,axisY4, 'bo-')
    elif(graphType == 'bar'):
        graphREC.bar(axisX4,axisY4, align='center')
        graphREC.xaxis_date()
    graphREC.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX4,rotation=90)
    graphREC.grid(True)

    graphSSA.set_title("SALVADOR")
    graphSSA.set_ylabel("Quantidade")
    graphSSA.set_xlabel("Dias")
    if (graphType == 'line'):
        graphSSA.plot(axisX5,axisY5, 'bo-')
    elif(graphType == 'bar'):
        graphSSA.bar(axisX5,axisY5, align='center')
        graphSSA.xaxis_date()
    graphSSA.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX5,rotation=90)
    graphSSA.grid(True)

    graphRIO.set_title("RIO DE JANEIRO")
    graphRIO.set_ylabel("Quantidade")
    graphRIO.set_xlabel("Dias")
    if (graphType == 'line'):
        graphRIO.plot(axisX6,axisY6, 'bo-')
    elif(graphType == 'bar'):
        graphRIO.bar(axisX6,axisY6, align='center')
        graphRIO.xaxis_date()
    graphRIO.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX6,rotation=90)
    graphRIO.grid(True)

    graphBSB.set_title("BRASILIA")
    graphBSB.set_ylabel("Quantidade")
    graphBSB.set_xlabel("Dias")
    if (graphType == 'line'):
        graphBSB.plot(axisX7,axisY7, 'bo-')
    elif(graphType == 'bar'):
        graphBSB.bar(axisX7,axisY7, align='center')
        graphBSB.xaxis_date()
    
    graphBSB.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX7,rotation=90)
    graphBSB.grid(True)

    graphBHZ.set_title("BELO HORIZONTE")
    graphBHZ.set_ylabel("Quantidade")
    graphBHZ.set_xlabel("Dias")
    if (graphType == 'line'):
        graphBHZ.plot(axisX8,axisY8, 'bo-')
    elif(graphType == 'bar'):
        graphBHZ.bar(axisX8,axisY8, align='center')
        graphBHZ.xaxis_date()
    
    graphBHZ.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX8,rotation=90)
    graphBHZ.grid(True)

    plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
    plt.show()
    #plt.savefig("plot.png",dpi=96)


def plot_hist(axisX,axisY):

    ax = plt.subplot(111)
    ax.bar(axisX,axisY,align='center')
    plt.xticks(axisX,rotation=90)
    ax.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    ax.xaxis_date()

    plt.show()
    

def get_all_incidents_from_DB(inicioCopaConf,terminoCopaConf):

    clusteringDefault = '%Y/%m/%d'
    #clusterPorHora = '%Y/%B/%d %H:%m:%S'
    incidents = Incident.get_all()
    incidentsDates = [
                        (i.reporting_date).strftime(clusteringDefault) for i in incidents 
                        if (inicioCopaConf <= i.reporting_date) and (i.reporting_date <=terminoCopaConf)
                    ]
    
    return incidentsDates

def get_all_actions_from_DB(inicioCopaConf,terminoCopaConf):

    clusteringDefault = '%Y/%m/%d'
    #clusterPorHora = '%Y/%B/%d %H:%m:%S'
    actions = Sincronizacao.get_all()
    actionsDates = [
                        (i.data_hora_de_referencia).strftime(clusteringDefault) for i in actions
                        if (inicioCopaConf <= i.inicio) and (i.fim <=terminoCopaConf)
                    ]
    
    return actionsDates

def get_incidents_by_cop_from_DB(inicioCopaConf,terminoCopaConf,cop):

    clusteringDefault = '%Y/%m/%d'
    #clusterPorHora = '%Y/%B/%d %H:%m:%S'
    incidents = Incident.get_all()
    incidentsDates = [
                        (i.reporting_date).strftime(clusteringDefault) for i in incidents 
                        if (inicioCopaConf <= i.reporting_date) and (i.reporting_date <=terminoCopaConf and i['operations_center']==cop)
                    ]
    
    return incidentsDates

def get_all_incidentes_from_file(inicioCopaConf, terminoCopaConf):

    datesFromFile = read_file('clusterizado.c2')
    dates = map(date_converter, datesFromFile)
    incidentes = map(qtd_incidents, datesFromFile)
    datesIncidents = {}
    for i, v in enumerate(dates):
        tmpDate = datetime.strptime(v,"%Y/%B/%d")
        if (inicioCopaConf <= tmpDate) and (tmpDate <= terminoCopaConf):

            datesIncidents[time.strftime("%Y/%m/%d",time.strptime(v,"%Y/%B/%d"))] = incidentes[i]

    return datesIncidents

def get_COP_incidents(inicioCopaConf, terminoCopaConf):

    incidentes = Incident.get_all()
    cops = [
            incidente['operations_center'] for incidente in incidentes
            if (inicioCopaConf <= incidente.reporting_date) and (incidente.reporting_date <=terminoCopaConf)
            ]
    
    return collections.Counter(cops)    
    
def get_statics(dataArray):

    print (dataArray)
    sizeData, (minimum,maximum),arithmeticMean,variance,skeness,kurtosis = stats.describe(map(float,dataArray))
    #mean = np.mean(np.array(map(float,dataArray)))
    mean = np.mean(map(float,dataArray))
    vara = np.var(map(float,dataArray))

    print arithmeticMean,variance
    print mean, vara

    print "Correlacao: ",stats.pearsonr(map(float,dataArray),map(float,dataArray))

if __name__ == "__main__":
    
    inicioCopaConf = datetime(2013,6,10)
    terminoCopaConf = datetime(2013,7,3)


    # para recuperar do banco
    
    incidentsDates =  get_all_incidents_from_DB(inicioCopaConf,terminoCopaConf)
    dictionaryIncidentsDates = {}
    #Faz a contagem de incidentes por data segundo o padrão de cluster
    clusterIncidentes = collections.Counter(incidentsDates)

    for i,v in enumerate(incidentsDates):
        dictionaryIncidentsDates[v]=clusterIncidentes[v]
    
    #para recuperar de um arquivo já clusterizado
    #dictionaryIncidentsDates = get_all_incidentes_from_file(inicioCopaConf,terminoCopaConf)
    
    sortedIncidentDates = sorted(set(dictionaryIncidentsDates))

    axisX = []
    axisY = []
    for key in sortedIncidentDates:
        axisX.append(datetime.strptime(key,"%Y/%m/%d"))
        axisY.append(dictionaryIncidentsDates[key])
    
    get_statics(axisY)
    #plot_hist(axisX,axisY)

    #cops = get_COP_incidents(inicioCopaConf,terminoCopaConf)

    axisYbyCops = {}
    axisXbyCops = {}

    """
    for cop in cops.keys():
        axisYbyCops[cop]=[]
        axisXbyCops[cop]=[]
        incidentsDates =  get_incidents_by_cop_from_DB(inicioCopaConf,terminoCopaConf,cop)
        dictionaryIncidentsDates = {}
        #Faz a contagem de incidentes por data segundo o padrão de cluster
        clusterIncidentes = collections.Counter(incidentsDates)
        for i,v in enumerate(incidentsDates):
            dictionaryIncidentsDates[v]=clusterIncidentes[v]
        sortedIncidentDates = sorted(set(dictionaryIncidentsDates)) 
        for key in sortedIncidentDates:
            axisXbyCops[cop].append(datetime.strptime(key,"%Y/%m/%d"))
            axisYbyCops[cop].append(dictionaryIncidentsDates[key])                  

    plot_total('line',
                axisX,axisY,
                axisXbyCops['COC'],axisYbyCops['COC'],
                axisXbyCops['CC2 - FTC - SSA'],axisYbyCops['CC2 - FTC - SSA'],
                axisXbyCops['CCDA - FOR'],axisYbyCops['CCDA - FOR'],
                axisXbyCops['CCDA - REC'],axisYbyCops['CCDA - REC'],
                axisXbyCops['CCDA - SSA'],axisYbyCops['CCDA - SSA'],
                axisXbyCops['CCDA - RIO'],axisYbyCops['CCDA - RIO'],
                axisXbyCops['CCDA - BSB'],axisYbyCops['CCDA - BSB'],
                axisXbyCops['CCDA - BHZ'],axisYbyCops['CCDA - BHZ']                
                )
    """
    
    actionsDates =  get_all_actions_from_DB(inicioCopaConf,terminoCopaConf)
    print collections.Counter(actionsDates)
    