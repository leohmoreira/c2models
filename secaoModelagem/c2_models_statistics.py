# -*- coding: utf-8 -*- 
from math import radians, cos, sin, asin, sqrt
from scipy.misc import factorial
from datetime import datetime
from scipy.optimize import curve_fit
import math
import calendar
import collections
import argparse
import numpy as np
from numpy import array
from scipy import special, optimize
import matplotlib.pyplot as plt
import time
from matplotlib.dates import DateFormatter
from scipy import stats
from pylab import text,title
import os, sys
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.vq import vq, kmeans, whiten
#lib_path_Pacificador = os.path.abspath('/home/moreira/Projetos/COP/pacificador_cop')
lib_path_Pacificador = os.path.abspath('/opt/pacificador_cop/')
sys.path.append(lib_path_Pacificador)
from incidentes.models import *

#Constantes
dateDistanceLimit = 43200 #(12 horas em segundos)
actionSize = 43200 #(12 horas em segundos)
punctualActionSize = 0 #(1 hora em segundos)
inicioAmostragem = datetime(2013,6,10,0,0,0)
terminoAmostragem = datetime(2013,6,30,23,59,59)
#COPs avaliados
allCops = ['CCDA - BHZ',
            'CCDA - BSB',
            'CCDA - FOR',
            'CCDA - REC',
            'CCDA - RIO',
            'CCDA - SSA',
            'CC2 - FTC - SSA',
            'CCTI - SSA']
#localização do COPs
latLongCops={} # (latitude,longitude)
latLongCops['CC2 - FTC - SSA'] = (-12.97974,-38.48362)
latLongCops['CCDA - BHZ'] = (-19.88866,-43.93903)
latLongCops['CCDA - BSB'] = (-15.79388,-47.88271)
latLongCops['CCDA - FOR'] = (-3.7889,-38.5193)
latLongCops['CCDA - REC'] = (-8.046,-34.937)
latLongCops['CCDA - RIO'] = (-22.90597,-43.21631)
latLongCops['CCDA - SSA'] = (-12.97974,-38.48362)

#paleta com 25 cores
cores = ['#f0f8ff',#ALICE BLUE
         '#00ffff',#AQUA
         '#ffe4c4',#BISQUE
         '#000000',#BLACK
         '#0000ff',#BLUE
         '#8a2be2',#BLUEVIOLET
         '#a52a2a',#BROWN
         '#7fff00',#CHARTREUSE
         '#d2691e',#CHOCOLATE
         '#dc143c',#CRISMSON
         '#a9a9a9',#DARKGRAY
         '#006400',#DARKGREEN
         '#8b008b',#DARKMAGENTA
         '#8b0000',#DARKRED
         '#2f4f4f',#DARKSLATEGRAY
         '#ff00ff',#FUCHSIA
         '#daa520',#GOLD
         '#7cfc00',#LAWNGREEN
         '#ffb6c1',#LIGHTPINK
         '#808000',#PINK
         '#cd853f',#PERU
         '#f4a460',#SANDYBROWN
         '#708090',#SLATEGRAY
         '#ff6347',#TOMATO
         '#9acd32'#YELLOWGREEN
        ]

def get_dict_all_actions():
    """
        Retorna todas as ações agrupadas em um dicionário cuja chave é o nome do COP
    """
    allSincronizations = Sincronizacao.get_all()
    dictionaryAllActions = {}
    #allCops = get_all_cops()
    for cop in allCops:
        dictionaryAllActions[cop] = []
    dictionaryAllActions['TODOS'] = []

    # é necessário percorrer todas sincronizacoes ao inves de usar get_all_actions porque só sinc tem o id do COP
    for sinc in allSincronizations:
        for action in sinc.acoes:
            if (
                (sinc.cop_responsavel['id'] in allCops) and 
                (
                    ((action.tipo == 'PONTUAL') and (action.inicio >= inicioAmostragem) and (action.inicio <= terminoAmostragem)) or
                    ((action.tipo == 'INTERVALO') and (action.inicio >= inicioAmostragem and action.fim <= terminoAmostragem))
                )
            ):
                dictionaryAllActions['TODOS'].append(action)
                dictionaryAllActions[sinc.cop_responsavel['id']].append(action)
                                
    return dictionaryAllActions

def get_dict_all_actions_by_type(actionType):
    """
        Retorna todas as ações do tipo action Type agrupadas em um dicionário cuja chave é o nome do COP
    """
    allSincronizations = Sincronizacao.get_all()
    dictionaryAllActions = {}
    #allCops = get_all_cops()
    for cop in allCops:
        dictionaryAllActions[cop] = []
    dictionaryAllActions['TODOS'] = []

    # é necessário percorrer todas sincronizacoes ao inves de usar get_all_actions porque só sinc tem o id do COP
    for sinc in allSincronizations:
        for action in sinc.acoes:
            if (
                (actionType == 'PONTUAL') and 
                (sinc.cop_responsavel['id'] in allCops) and 
                (
                    ((action.tipo == 'PONTUAL') and (action.inicio >= inicioAmostragem) and (action.inicio <= terminoAmostragem))
                )
            ):
                dictionaryAllActions['TODOS'].append(action)
                dictionaryAllActions[sinc.cop_responsavel['id']].append(action)
            if (
                (actionType == 'INTERVALO') and 
                (sinc.cop_responsavel['id'] in allCops) and 
                (
                    ((action.tipo == 'INTERVALO') and (action.inicio >= inicioAmostragem and action.fim <= terminoAmostragem))  
                )
            ):
                dictionaryAllActions['TODOS'].append(action)
                dictionaryAllActions[sinc.cop_responsavel['id']].append(action)
                                
    return dictionaryAllActions

def get_all_actions():
    """
        Retorna todas as ações agrupadas em um array
    """

    allSincronizations = Sincronizacao.get_all()
    allActions = []
    #allCops = get_all_cops()
    for sinc in allSincronizations:
        for action in sinc.acoes:
            if (
                (sinc.cop_responsavel['id'] in allCops) and 
                (
                    ((action.tipo == 'PONTUAL') and (action.inicio >= inicioAmostragem) and (action.inicio <= terminoAmostragem)) or
                    ((action.tipo == 'INTERVALO') and (action.inicio >= inicioAmostragem and action.fim <= terminoAmostragem))
                )
            ):
                allActions.append(action)
        
    return allActions

def get_actions_near_date(listActions,date, mask = '%Y/%m/%d'):
    """
    Retorna todas as ações de "listActions" nas quais o valor "date" seja igual, segundo "MASK", ao inicio da ação
    Por padrão, MASK considera apenas ANO/MES/DIA, com hora 00:00:00
    """
    
    return [action for action in listActions
        if datetime.strptime(datetime.strftime(action.inicio,mask),mask) == datetime.strptime(datetime.strftime(date,mask),mask)
    ]

def get_all_incidents():
    """
        Retorna todos os incidentes agrupados em um array
    """
    allIncidents = Incident.get_all()
    #allCops = get_all_cops()
    incidents = []
    for i in allIncidents:
                        if(
                            (i['operations_center'] in allCops) and
                            (inicioAmostragem <= i.reporting_date and i.reporting_date <=terminoAmostragem)
                        ):
                        # transformando CCTI - SSA e CC2 - FTC - SSA em CCDA - SSA
                            if(i['operations_center'] == 'CCTI - SSA' or i['operations_center'] == 'CC2 - FTC - SSA'):
                                i['operations_center'] = 'CCDA - SSA'
                            incidents.append(i)
    return incidents    
    
def get_dict_all_incidents():
    """
        Retorna todos os incidentes agrupados em um dicionário cuja chave é o nome do COP
    """
    dictionaryAllIncidents = {}
    #allCops = get_all_cops()
    
    for cop in allCops:
        dictionaryAllIncidents[cop] = []
    dictionaryAllIncidents['TODOS'] = []

    allIncidents = get_all_incidents()
    for incident in allIncidents:
        dictionaryAllIncidents['TODOS'].append(incident)
        dictionaryAllIncidents[incident['operations_center']].append(incident)
                
    return dictionaryAllIncidents


def get_incidents_near_date(listIncidents,date,mask = '%Y/%m/%d'):
    """
    Retorna todas os incidentes de "listIncidentes" nas quais o valor "date" seja igual, segundo "MASK", ao reporting_date
    Por padrão, MASK considera apenas ANO/MES/DIA, com hora 00:00:00
    """
    return [incident for incident in listIncidents
            if datetime.strptime(datetime.strftime(incident.reporting_date,mask),mask) == datetime.strptime(datetime.strftime(date,mask),mask)
    ]

def get_all_reports():
    """
        Retorna todos os relatos de situação agrupados em um array
    """
    allReports = RelatoDeSituacao.get_all()
    #allCops = get_all_cops()
    reports = []
    for r in allReports:
            if (
                    inicioAmostragem <= r.data_hora and 
                    r.data_hora <=terminoAmostragem and
                    'cop' in r.relator and # todos tem que ter o COP
                    r.relator['cop'] in allCops
                    #r.relator['cop'] != 'COC' # desconsiderei COC
                ):
                    # transformando CCTI - SSA e CC2 - FTC - SSA em CCDA - SSA
                    if(r.relator['cop'] == 'CCTI - SSA' or r.relator['cop'] == 'CC2 - FTC - SSA'):
                        r.relator['cop'] = 'CCDA - SSA'
                    reports.append(r)
    return reports
        
def get_dict_all_reports():
    """
        Retorna todos os relatos agrupados em um dicionário cuja chave é o nome do COP
    """
    dictionaryAllReports = {}
    #allCops = get_all_cops()
    
    for cop in allCops:
        dictionaryAllReports[cop] = []
    dictionaryAllReports['TODOS'] = []

    allReports = get_all_reports()
    for report in allReports:
        dictionaryAllReports['TODOS'].append(report)
        dictionaryAllReports[report.relator['cop']].append(report)
                
    return dictionaryAllReports

def get_reports_near_date(listReports,date,mask = '%Y/%m/%d'):
    """
    Retorna todas os relatos de situação de "listReports" nas quais o valor "date" seja igual, segundo "MASK", ao data_hora
    Por padrão, MASK considera apenas ANO/MES/DIA, com hora 00:00:00
    """
    return [report for report in listReports
            if datetime.strptime(datetime.strftime(report.data_hora,mask),mask) == datetime.strptime(datetime.strftime(date,mask),mask)
    ]

def plot_graph(filename,title,axisX,serie1,serie2,rotulos):

    fig, graph = plt.subplots()
    graph.set_title(title + " - Correlacao: " + str(stats.pearsonr(serie1,serie2)[0]))
    graph.set_ylabel("Quantidade")
    graph.set_xlabel("Dias")
    lineObjects = graph.plot(axisX,serie1, 'ro-',axisX,serie2, 'bo-')
    graph.xaxis_date()
    graph.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graph.grid(True)
    plt.legend(iter(lineObjects),(rotulos), borderaxespad=0, bbox_to_anchor=(1.11, 0.5),prop={'size':12})
    fig.set_size_inches(18.5,10.5)
    fig.savefig(filename,dpi=96)

def plot_total(filename,axisX,
                incidentsTotal,actionsTotal,
                incidents1,actions1,
                incidents2,actions2,
                incidents3,actions3,
                incidents4,actions4,
                incidents5,actions5,
                incidents6,actions6
            ):

    plt.close('all')
    fig = plt.figure()

    graphTotal = plt.subplot2grid((4,2),(0,0),colspan=2)      
    graphTotal.set_title("Total - Correlacao: " + str(stats.pearsonr(incidentsTotal,actionsTotal)[0]))
    graphTotal.set_ylabel("Quantidade")
    graphTotal.set_xlabel("Dias")
    graphTotal.plot(axisX,incidentsTotal, 'ro-',axisX,actionsTotal, 'bo-')
    graphTotal.xaxis_date()
    graphTotal.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphTotal.grid(True)

    graphRIO = plt.subplot2grid((4,2),(1,0))      
    graphRIO.set_title("RIO DE JANEIRO - Correlacao: " + str(stats.pearsonr(incidents1,actions1)[0]))
    graphRIO.set_ylabel("Quantidade")
    graphRIO.set_xlabel("Dias")
    graphRIO.plot(axisX,incidents1, 'ro-',axisX,actions1, 'bo-')
    graphRIO.xaxis_date()
    graphRIO.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphRIO.grid(True)

    graphBSB = plt.subplot2grid((4,2),(1,1))
    graphBSB.set_title("BRASILIA - Correlacao: " + str(stats.pearsonr(incidents2,actions2)[0]))
    graphBSB.set_ylabel("Quantidade")
    graphBSB.set_xlabel("Dias")
    graphBSB.plot(axisX,incidents2, 'ro-',axisX,actions2, 'bo-')
    graphBSB.xaxis_date()
    graphBSB.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphBSB.grid(True)  

    graphSSA = plt.subplot2grid((4,2),(2,0))
    graphSSA.set_title("SALVADOR - Correlacao: " + str(stats.pearsonr(incidents3,actions3)[0]))
    graphSSA.set_ylabel("Quantidade")
    graphSSA.set_xlabel("Dias")
    graphSSA.plot(axisX,incidents3, 'ro-',axisX,actions3, 'bo-')
    graphSSA.xaxis_date()
    graphSSA.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphSSA.grid(True)  

    graphREC = plt.subplot2grid((4,2),(2,1))
    graphREC.set_title("RECIFE - Correlacao: " + str(stats.pearsonr(incidents4,actions4)[0]))
    graphREC.set_ylabel("Quantidade")
    graphREC.set_xlabel("Dias")
    graphREC.plot(axisX,incidents4, 'ro-',axisX,actions4, 'bo-')
    graphREC.xaxis_date()
    graphREC.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphREC.grid(True)  

    graphFOR = plt.subplot2grid((4,2),(3,0))
    graphFOR.set_title("FORTALEZA - Correlacao: " + str(stats.pearsonr(incidents5,actions5)[0]))
    graphFOR.set_ylabel("Quantidade")
    graphFOR.set_xlabel("Dias")
    graphFOR.plot(axisX,incidents5, 'ro-',axisX,actions5, 'bo-')
    graphFOR.xaxis_date()
    graphFOR.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphFOR.grid(True)

    graphBHZ = plt.subplot2grid((4,2),(3,1))
    graphBHZ.set_title("BELO HORIZONTE - Correlacao: " + str(stats.pearsonr(incidents6,actions6)[0]))
    graphBHZ.set_ylabel("Quantidade")
    graphBHZ.set_xlabel("Dias")
    graphBHZ.plot(axisX,incidents6, 'ro-',axisX,actions6, 'bo-')
    graphBHZ.xaxis_date()
    graphBHZ.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphBHZ.grid(True)    

    plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
    #plt.show()
    fig.set_size_inches(18.5,10.5)
    fig.savefig(filename,dpi=96)

def plot_graph_bar(filename,title,axisX,axisY,rotulo,barColor):

    fig, graph = plt.subplots()
    graph.set_title(title)
    graph.set_ylabel("Quantidade")
    graph.set_xlabel("Dias")
    plt.bar(axisX,axisY,color=barColor,label = rotulo, align='center')
    graph.xaxis_date()
    graph.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graph.grid(True)
    plt.legend()
    sizeData, (minimum,maximum),arithmeticMean,variance,skeness,kurtosis = stats.describe(axisY)
    fig.text(.91,.52,"Media = " + str(arithmeticMean),fontsize=10)
    fig.text(.91,.5,"Minimo = " + str(minimum),fontsize=10)
    fig.text(.91,.48,"Maximo = " + str(maximum),fontsize=10)
    fig.text(.91,.46,"Variancia = " + str('%.2f' % round(variance,2)),fontsize=10)
    fig.set_size_inches(18.5,10.5)
    fig.savefig(filename,dpi=96)

def dateChangeFormat(item):

    return datetime.strftime(item,"%d/%m")

def plot_graph_bar_full(filename,title,axisX,
                            serie1,rotulo1,barColor1,
                            serie2,rotulo2,barColor2,
                            serie3,rotulo3,barColor3,
                            serie4,rotulo4,barColor4,
                            serie5,rotulo5,barColor5,
                            serie6,rotulo6,barColor6,
                            serie7,rotulo7,barColor7,
                            serie8,rotulo8,barColor8):

    barWidth = 0.1
    valuesAxisX = map(dateChangeFormat,axisX)
    index = np.arange(len(axisX))
    fig, graph = plt.subplots()
    graph.set_title(title)
    graph.set_ylabel("Quantidade")
    graph.set_xlabel("Dias")
    plt.bar(index,serie1,barWidth,color=barColor1,label = rotulo1)
    plt.bar(index+barWidth,serie2,barWidth,color=barColor2,label = rotulo2)
    plt.bar(index+2*barWidth,serie3,barWidth,color=barColor3,label = rotulo3)
    plt.bar(index+3*barWidth,serie4,barWidth,color=barColor4,label = rotulo4)
    plt.bar(index+4*barWidth,serie5,barWidth,color=barColor5,label = rotulo5)
    plt.bar(index+5*barWidth,serie6,barWidth,color=barColor6,label = rotulo6)
    plt.bar(index+6*barWidth,serie7,barWidth,color=barColor7,label = rotulo7)
    plt.bar(index+7*barWidth,serie8,barWidth,color=barColor8,label = rotulo8)
    graph.xaxis_date()
    graph.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(index + barWidth,valuesAxisX)
    graph.grid(True)
    plt.legend()
    #sizeData, (minimum,maximum),arithmeticMean,variance,skeness,kurtosis = stats.describe(axisY)
    #fig.text(.91,.52,"Media = " + str(arithmeticMean),fontsize=10)
    #fig.text(.91,.5,"Minimo = " + str(minimum),fontsize=10)
    #fig.text(.91,.48,"Maximo = " + str(maximum),fontsize=10)
    #fig.text(.91,.46,"Variancia = " + str('%.2f' % round(variance,2)),fontsize=10)
    fig.set_size_inches(18.5,10.5)
    fig.savefig(filename,dpi=96)

def plot_resume_cop(filename,cop,axisX,actions,incidents,reports):

    plt.close('all')
    fig = plt.figure()

    graphIncidentsActions = plt.subplot2grid((3,1),(0,0))      
    graphIncidentsActions.set_title(cop+"\nIncidentes & Acoes - Correlacao: " + str(stats.pearsonr(actions,incidents)[0]))
    graphIncidentsActions.set_ylabel("Quantidade")
    graphIncidentsActions.set_xlabel("Dias")
    linesIncidentsActions = graphIncidentsActions.plot(axisX,incidents, 'ro-',axisX,actions, 'bo-')
    graphIncidentsActions.xaxis_date()
    graphIncidentsActions.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphIncidentsActions.grid(True)
    plt.legend(iter(linesIncidentsActions),('Incidentes','Acoes'),prop={'size':10},bbox_to_anchor=(1, 1.4))

    graphReportsActions = plt.subplot2grid((3,1),(1,0))      
    graphReportsActions.set_title(cop+"\nRelatos & Acoes - Correlacao: " + str(stats.pearsonr(actions,reports)[0]))
    graphReportsActions.set_ylabel("Quantidade")
    graphReportsActions.set_xlabel("Dias")
    linesReportsActions = graphReportsActions.plot(axisX,reports, 'go-',axisX,actions, 'bo-')
    graphReportsActions.xaxis_date()
    graphReportsActions.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphReportsActions.grid(True)
    plt.legend(iter(linesReportsActions),('Relatos','Acoes'),prop={'size':10},bbox_to_anchor=(1, 1.4))#, borderaxespad=0, bbox_to_anchor=(1.11, 0.5),prop={'size':12})

    # correlacao entre incidentes e relatos
    """
    graphReportsIncidents = plt.subplot2grid((4,1),(2,0))      
    graphReportsIncidents.set_title(cop+"\nRelatos & Incidentes - Correlacao: " + str(stats.pearsonr(incidents,reports)[0]))
    graphReportsIncidents.set_ylabel("Quantidade")
    graphReportsIncidents.set_xlabel("Dias")
    linesReportsIncidents = graphReportsIncidents.plot(axisX,reports, 'go-',axisX,incidents, 'ro-')
    graphReportsIncidents.xaxis_date()
    graphReportsIncidents.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphReportsIncidents.grid(True)
    plt.legend(iter(linesReportsIncidents),('Relatos','Incidentes'),prop={'size':10})
    """     
    # correlacao entre acoes e (incidentes + relatos)

    incRel = [i+r for i,r in zip(incidents,reports)]

    graphIncRelsActions = plt.subplot2grid((3,1),(2,0))      
    graphIncRelsActions.set_title(cop+"\nIncidentes + Relatos & Acoes - Correlacao: " + str(stats.pearsonr(actions,incRel)[0]))
    graphIncRelsActions.set_ylabel("Quantidade")
    graphIncRelsActions.set_xlabel("Dias")
    linesIncRelsActions = graphIncRelsActions.plot(axisX,incRel, 'co-',axisX,actions, 'bo-')
    graphIncRelsActions.xaxis_date()
    graphIncRelsActions.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphIncRelsActions.grid(True)
    plt.legend(iter(linesIncRelsActions),('Incidentes + Relatos','Acoes'),prop={'size':10},bbox_to_anchor=(1, 1.4))

    plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
    fig.set_size_inches(18.5,10.5)
    fig.savefig(filename,dpi=96)

"""
def plot_resume_cop(filename,cop,axisX,incidents,actions,punctualActions,intervalActions):

    plt.close('all')
    fig = plt.figure()

    graphIncidentsActions = plt.subplot2grid((3,2),(0,0),colspan=2)      
    graphIncidentsActions.set_title(cop+"\nIncidentes & Acoes - Correlacao: " + str(stats.pearsonr(incidents,actions)[0]))
    graphIncidentsActions.set_ylabel("Quantidade")
    graphIncidentsActions.set_xlabel("Dias")
    linesIncidentsActions = graphIncidentsActions.plot(axisX,incidents, 'ro-',axisX,actions, 'bo-')
    graphIncidentsActions.xaxis_date()
    graphIncidentsActions.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphIncidentsActions.grid(True)
    plt.legend(iter(linesIncidentsActions),('Incidentes','Acoes'),prop={'size':10})#, borderaxespad=0, bbox_to_anchor=(1.11, 0.5),prop={'size':12})

    graphIncidents = plt.subplot2grid((3,2),(1,0))      
    graphIncidents.set_title("Incidentes")
    graphIncidents.set_ylabel("Quantidade")
    graphIncidents.set_xlabel("Dias")
    graphIncidents.plot(axisX,incidents, 'ro-')
    graphIncidents.xaxis_date()
    graphIncidents.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphIncidents.grid(True)

    graphActions = plt.subplot2grid((3,2),(1,1))
    graphActions.set_title("Acoes")
    graphActions.set_ylabel("Quantidade")
    graphActions.set_xlabel("Dias")
    graphActions.plot(axisX,actions, 'bo-')
    graphActions.xaxis_date()
    graphActions.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphActions.grid(True)  

    graphPunctualActions = plt.subplot2grid((3,2),(2,0))
    graphPunctualActions.set_title("Acoes Pontuais")
    graphPunctualActions.set_ylabel("Quantidade")
    graphPunctualActions.set_xlabel("Dias")
    graphPunctualActions.plot(axisX,punctualActions, 'go-')
    graphPunctualActions.xaxis_date()
    graphPunctualActions.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphPunctualActions.grid(True)  

    graphIntervalActions = plt.subplot2grid((3,2),(2,1))
    graphIntervalActions.set_title("Acoes Intervalo")
    graphIntervalActions.set_ylabel("Quantidade")
    graphIntervalActions.set_xlabel("Dias")
    graphIntervalActions.plot(axisX,intervalActions, 'co-')
    graphIntervalActions.xaxis_date()
    graphIntervalActions.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphIntervalActions.grid(True)  
    
    plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
    fig.set_size_inches(18.5,10.5)
    fig.savefig(filename,dpi=96)
"""

def plot_graph_pie(filename,titulo,serie):

    """
        Constroi um grafico pizza
    """
    validSerie = []
    labels =[]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','cyan','green','pink']
    explode = [0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    for cop in serie.keys():
        if cop != 'TODOS' :
            validSerie.append(cop)
            labels.append(cop)
    sizes = []
    for cop in validSerie:
        sizes.append(len(serie[cop]))

    plt.close('all')
    title(titulo)
    plt.pie(sizes,labels=labels,colors=colors,explode=explode,autopct='%1.1f%%')
    plt.axis('equal')
    plt.savefig(filename,dpi=96)

def graph_incidents_per_action(cop,incidents,actions):

    incAction = {}
    for i, a in zip(incidents,actions):
        incAction[a] = i
    tmpAction =[]
    tmpInc =[]
    tmpNew = []
    for a in sorted(incAction):
        tmpAction.append(a)
        tmpInc.append(incAction[a])
    print cop, " === ",tmpAction
    #popt, pocv = curve_fit(func,tmpAction,tmpInc)
    #print popt
    plt.close('all')
    plt.plot(tmpAction,tmpInc,'ro-')#,tmpAction,func(tmpAction,popt[0],popt[1],popt[2],popt[3]),'g^-')
    plt.grid(True)
    plt.savefig('qtdeIncxQtdAccoes_'+cop+'.png',dpi=96)
    #plt.show()

def funcExpoPoisson(x,min,max):

    return 1000 / (x * (np.log(max) - np.log(min)))

def funcExpGenLinear(x,a,b,c,d,e,f):

    return a * (b**(c*x)) + d * (e**(f*x))   
    
def compute_statistics(serie):
    """
        Computa as estatísticas de SERIE utilizando stats.describe
    """
    sizeData, (minimum,maximum),arithmeticMean,variance,skeness,kurtosis = stats.describe(serie)

    print "Size Data  = ",sizeData , "Minimo,Maximo = ",(minimum,maximum), "Média = ", arithmeticMean , "Variância = ", variance

def interArrrival_time_distribution(filename,cop,serie, nbins=30,limit = 24*3600,cor='green'):

    """
        Calcula a distribuição dos tempos entre ocorrencias dos incidentes.
        Salva em arquivo
    """
    arrivalTime = []
    
    for i in serie:
        if (hasattr(i,'reporting_date')): # é incidente
            arrivalTime.append(datetime.strptime(datetime.strftime(i.reporting_date,"%Y-%m-%d %H:%M:%S"),"%Y-%m-%d %H:%M:%S"))
        elif (hasattr(i,'data_hora')): # é relato
            arrivalTime.append(datetime.strptime(datetime.strftime(i.data_hora,"%Y-%m-%d %H:%M:%S"),"%Y-%m-%d %H:%M:%S"))
    sortedArrivalTime =  sorted(arrivalTime)

    interArrivalTime = []
    if(len(sortedArrivalTime)>0):
        for i in range(0,len(sortedArrivalTime)-1):
            interArrivalTime.append((sortedArrivalTime[i+1] - sortedArrivalTime[i]).total_seconds())
    
    plt.close('all')
    fig = plt.figure()

    if(len(interArrivalTime)>0):
       
        qtde, bins, patches = plt.hist(interArrivalTime, nbins, range=(0,limit),facecolor=cor, alpha=0.5)
        poptLinear, pocvLinear = curve_fit(funcExpGenLinear,np.array(bins[:-1]),np.array(qtde))
        plt.plot(bins[:-1],qtde,'ro-',
           bins[:-1],funcExpGenLinear(np.array(bins[:-1]),*poptLinear),'b^-')
        
        fig.suptitle(cop+"\nIntervalo de tempo em ocorrencias sequenciais")
        plt.ylabel("Probabilidade (%)")
        plt.xlabel("Intervalo (s)")
        plt.xticks(bins[:-1],rotation=45)
        plt.grid(True)
        fig.set_size_inches(18.5,10.5)
        fig.savefig(filename+cop+'.png',dpi=96)
        plt.close('all')

    
def interArrrival_distance_distribution(tipo,filename,cop,serie, nbins=30,limit = 10,cor='gray'):

    """
        Calcula a distribuição da distancia entre ocorrencias dos incidentes.
        Salva em arquivo
    """
    
    interArrivalDistance = []
    #ordena sequencialmente no tempo os incidentes
    #if(hasattr(serie[0],'reporting_date')):
    if(tipo=='incidentes'):
        arrivalSequence = sorted(serie,key=lambda x: x.reporting_date)
        if(len(arrivalSequence)>0):
            for i in range(0,len(arrivalSequence)-1):
                if(serie[i].lon and serie[i].lat and serie[i+1].lon and serie[i+1].lat):
                    interArrivalDistance.append(haversine(
                        float(serie[i+1].lon),float(serie[i+1].lat),
                        float(serie[i].lon),float(serie[i].lat)
                    ))
    elif(tipo=='relatos'):#(hasattr(serie[0],'data_hora')): # é relato)
        arrivalSequence = sorted(serie,key=lambda x: x.data_hora)
        if(len(arrivalSequence)>0):
            for i in range(0,len(arrivalSequence)-1):
                #if(serie[i].lon and serie[i].lat and serie[i+1].lon and serie[i+1].lat):
                if(hasattr(serie[i].localizacao,'lon') and 
                    hasattr(serie[i].localizacao,'lat') and
                    hasattr(serie[i+1].localizacao,'lon') and 
                    hasattr(serie[i+1].localizacao,'lat')
                    ):
                    interArrivalDistance.append(haversine(
                        float(serie[i+1].localizacao['lon']),float(serie[i+1].localizacao['lat']),
                        float(serie[i].localizacao['lon']),float(serie[i].localizacao['lat'])
                    ))

    plt.close('all')
    fig = plt.figure()
    if(len(interArrivalDistance)>0):
        qtde, bins, patches = plt.hist(interArrivalDistance, nbins, range=(0,limit),facecolor=cor, alpha=0.5)
        
        poptLinear, pocvLinear = curve_fit(funcExpGenLinear,np.array(bins[:-1]),np.array(qtde))

        plt.plot(bins[:-1],qtde,'ro-',
            bins[:-1],funcExpGenLinear(np.array(bins[:-1]),*poptLinear),'b^-')
        
        fig.suptitle(cop+"\nIntervalo de distancia das ocorrencias sequenciais")
        plt.xlabel("Distancia (km)")
        plt.ylabel("Probabilidade (%)")
        #plt.xticks(bins[:-1],rotation=90)
        plt.grid(True)
        fig.set_size_inches(18.5,10.5)
        fig.savefig(filename+cop+'.png',dpi=96)
        plt.close('all')
    
def incidents_location(filename,cop,serie):

    """
        Plota a localização dos incidentes long (eixo X) x lat (eixo Y)
    """
    
    lats = []
    longs = []
    tempo = []
    cluster2DLatLong =[]
    for i in serie:
        if(hasattr(i,'reporting_date')): # é incidentes
            tempo.append(i.reporting_date)
            if(i.lon and i.lat):
                cluster2DLatLong.append([float(i.lat),float(i.lon)])
            if(cop == 'TODOS' and i.lon and i.lat):
                lats.append(float(i.lat))
                longs.append(float(i.lon))
            elif(i.lon and i.lat and haversine(float(latLongCops[cop][1]),float(latLongCops[cop][0]),float(i.lon),float(i.lat))<=50):
                lats.append(float(i.lat))
                longs.append(float(i.lon))
        elif(hasattr(i,'data_hora') and hasattr(i.localizacao,'lat') and hasattr(i.localizacao,'lon')): # é relato
            tempo.append(i.data_hora)
        #    if(i.localizacao['lon'] and i.localizacao['lat']):
            cluster2DLatLong.append([float(i.localizacao['lat']),float(i.localizacao['lon'])])
            if(cop == 'TODOS'): #and i.localizacao['lon'] and i.localizacao['lat']):
                lats.append(float(i.localizacao['lat']))
                longs.append(float(i.localizacao['lon']))
            #elif(i.localizacao['lon'] and i.localizacao['lat'] and haversine(float(latLongCops[cop][1]),float(latLongCops[cop][0]),float(i.localizacao['lon']),float(i.localizacao['lat']))<=50):
            elif(haversine(float(latLongCops[cop][1]),float(latLongCops[cop][0]),float(i.localizacao['lon']),float(i.localizacao['lat']))<=50):
                lats.append(float(i.localizacao['lat']))
                longs.append(float(i.localizacao['lon']))

    if(len(lats)>0 and len(longs)>0):
        #centro de massa = media da latitude e longitude
        sizeData, (minimum,maximum),arithmeticMean,variance,skeness,kurtosis = stats.describe(lats)
        mediaLat = arithmeticMean
        sizeData, (minimum,maximum),arithmeticMean,variance,skeness,kurtosis = stats.describe(longs)
        mediaLon = arithmeticMean

        plt.close('all')
        fig = plt.figure()
        fig.suptitle(cop+"\nLocalizacao")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        fig.set_size_inches(18.5,10.5)
        plt.plot(longs,lats,"ro")
        plt.plot(mediaLon,mediaLat,'b*')
        plt.grid(True)
        fig.savefig(filename+cop+'.png',dpi=96)
        plt.close('all')
    
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

def computeCluster(filename,cop,serie):
    """
        Calcula a quantide de cluster necessaria (Rule o Thumbs) e o centros de cada
    """
    latitudes = []
    longitudes = []
    tempo = []
    cluster3DLatLong =[]
    serieItens =[]
    
    for i in serie:
        if(hasattr(i,'reporting_date')): # é incidentes
            if(cop == 'TODOS' and i.lon and i.lat):
                latitudes.append(float(i.lat))
                longitudes.append(float(i.lon))
                tempo.append(i.reporting_date)
                cluster3DLatLong.append([float(i.lat),float(i.lon)])
                serieItens.append(i)
                
            elif(i.lon and i.lat and haversine(float(latLongCops[cop][1]),float(latLongCops[cop][0]),float(i.lon),float(i.lat))<=50):
                latitudes.append(float(i.lat))
                longitudes.append(float(i.lon))
                tempo.append(i.reporting_date)
                cluster3DLatLong.append([float(i.lat),float(i.lon)])
                serieItens.append(i)
                
        elif(hasattr(i,'data_hora') and hasattr(i.localizacao,'lat') and hasattr(i.localizacao,'lon')): # é relato
            if(cop == 'TODOS' and hasattr(i.localizacao,'lat') and hasattr(i.localizacao,'lon')):
                latitudes.append(float(i.localizacao['lat']))
                longitudes.append(float(i.localizacao['lon']))
                tempo.append(i.data_hora)
                cluster3DLatLong.append([float(i.localizacao['lat']),float(i.localizacao['lon'])])
                serieItens.append(i)
                
            elif(hasattr(i.localizacao,'lat') and hasattr(i.localizacao,'lon') and
                haversine(float(latLongCops[cop][1]),float(latLongCops[cop][0]),float(i.localizacao['lon']),float(i.localizacao['lat']))<=50):
                latitudes.append(float(i.localizacao['lat']))
                longitudes.append(float(i.localizacao['lon']))
                tempo.append(i.data_hora)
                cluster3DLatLong.append([float(i.localizacao['lat']),float(i.localizacao['lon'])])
                serieItens.append(i)
                
    #clusters geograficos
    features  = array(zip(latitudes,longitudes))
    # escolhi pegar o maior valor menor q sqrt(n/2)
    #k = int(math.floor(math.sqrt(len(latitudes)/2.0)))
    k = int(math.floor(math.sqrt(len(latitudes)/4.0)))
    if (k==0): 
        k = 1
    clusters,distorcao = kmeans(features,k)
    
    #criando um vetor com a qtde de clusters necessarios
    itensClusterizados = []
    for i in range(0,k):
        itensClusterizados.append([])
    #agrupando cada item no seu cluster
    for i in range(0,len(tempo)):
        distancias=[]
        for c in clusters:
            #calcula a distancia o item ao centro de cada cluster
            distancias.append(haversine(float(longitudes[i]),float(latitudes[i]),float(c[1]),float(c[0])))
        #armazena o item no cluster mais proximo
        itensClusterizados[distancias.index(np.min(distancias))].append(serieItens[i])

    menorTempo = np.min(tempo)
    #criando os graficos ... cada grafico com uma cor
    
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors=['y','b','r']
    for c in range(0,k):
        for i in itensClusterizados[c]:
            if(hasattr(i,'reporting_date') and i.lon and i.lat): # é incidentes
                ax.scatter(float(i.lat), float(i.lon), (i.reporting_date - menorTempo).total_seconds(),c='r',s=100)
            elif(hasattr(i,'data_hora') and hasattr(i.localizacao,'lat') and hasattr(i.localizacao,'lon')): # é relato
                ax.scatter(float(i.localizacao['lat']), float(i.localizacao['lon']), (i.data_hora - menorTempo).total_seconds(),c='g',s=100)
    ax.set_title('Ocorrencias', fontsize=24)
    ax.set_xlabel('Latitude', fontsize=20)
    ax.set_ylabel('Longitude', fontsize=20)
    ax.set_zlabel('Tempo', fontsize=20)
    fig.set_size_inches(18.5,10.5)
    fig.savefig(filename+cop+'.png',dpi=96)
    
    return clusters,itensClusterizados

if __name__ == "__main__":
    """
        Loop principal
    """
    
    matchDays = [datetime(2013,6,15,0,0,0),datetime(2013,6,16,0,0,0),datetime(2013,6,17,0,0,0),datetime(2013,6,19,0,0,0),datetime(2013,6,20,0,0,0),
                 datetime(2013,6,22,0,0,0),datetime(2013,6,23,0,0,0),datetime(2013,6,26,0,0,0),datetime(2013,6,27,0,0,0),datetime(2013,6,30,0,0,0)]
    
    mdays = [#datetime(2013,6,10),datetime(2013,6,11),datetime(2013,6,12),datetime(2013,6,13),
            #datetime(2013,6,14),
            datetime(2013,6,15),datetime(2013,6,16),datetime(2013,6,17),datetime(2013,6,18),datetime(2013,6,19),
            datetime(2013,6,20),datetime(2013,6,21),datetime(2013,6,22),datetime(2013,6,23),datetime(2013,6,24),
            datetime(2013,6,25),datetime(2013,6,26),datetime(2013,6,27),datetime(2013,6,28),datetime(2013,6,29),datetime(2013,6,30)]
            #datetime(2013,7,1)]
    #matchDays = mdays
    # inicio da geracao dos dados para estatisticas
    allActionsDict = get_dict_all_actions()
    allPunctualActionsDict = get_dict_all_actions_by_type('PONTUAL')
    allIntervalActionsDict = get_dict_all_actions_by_type('INTERVALO')
    allIncidentsDict = get_dict_all_incidents()
    allReportsDict = get_dict_all_reports()
    allIncidentsReportsDict = {}

    #allCops = get_all_cops()
    incidentsSerie = {}
    actionsSerie = {}
    punctualActionsSerie = {}
    intervalActionsSerie = {}
    reportsSerie = {}
    incidentsReportsSerie = {}

    incidentsSerie['TODOS'] = []
    actionsSerie['TODOS'] = []
    punctualActionsSerie['TODOS'] = []
    intervalActionsSerie['TODOS'] = []
    reportsSerie['TODOS'] = []
    incidentsReportsSerie['TODOS'] = []
    for day in matchDays:
        #for day in mdays:
            incidentsSerie['TODOS'].append(len(get_incidents_near_date(allIncidentsDict['TODOS'],day)))
            actionsSerie['TODOS'].append(len(get_actions_near_date(allActionsDict['TODOS'],day)))
        #    punctualActionsSerie['TODOS'].append(len(get_actions_near_date(allPunctualActionsDict['TODOS'],day)))
        #    intervalActionsSerie['TODOS'].append(len(get_actions_near_date(allIntervalActionsDict['TODOS'],day)))
            reportsSerie['TODOS'].append(len(get_reports_near_date(allReportsDict['TODOS'],day)))
    
    # agrupar incidentes e relatos
    incidentsReportsSerie['TODOS'] = [i + r for i,r in zip(incidentsSerie['TODOS'],reportsSerie['TODOS'])]
    allIncidentsReportsDict['TODOS'] = allIncidentsDict['TODOS'] + allReportsDict['TODOS']

    for cop in allCops:
        incidentsSerie[cop]=[]
        actionsSerie[cop]=[]
        punctualActionsSerie[cop] = []
        intervalActionsSerie[cop] = []
        reportsSerie[cop] = []
        incidentsReportsSerie[cop] = []

        for day in matchDays:
            incidentsSerie[cop].append(len(get_incidents_near_date(allIncidentsDict[cop],day)))
            actionsSerie[cop].append(len(get_actions_near_date(allActionsDict[cop],day)))
            reportsSerie[cop].append(len(get_reports_near_date(allReportsDict[cop],day)))
            # agrupar incidentes e relatos
            incidentsReportsSerie[cop] = [i + r for i,r in zip(incidentsSerie[cop],reportsSerie[cop])]
            allIncidentsReportsDict[cop] = allIncidentsDict[cop] + allReportsDict[cop]
    
    # termino da geracao dos dados para estatisticas
    
    # inicio da criacao dos graficos

    # cluster 3D
    computeCluster('Cluster3D_IncidentesRelatos_','TODOS',allIncidentsReportsDict['TODOS'])
    # cluster de ocorrência de incidentes
    incidents_location('Localizacao_Incidentes_','TODOS',allIncidentsDict['TODOS']) # unidade em km
    # cluster de ocorrência de relatos
    incidents_location('Localizacao_Relatos_','TODOS',allReportsDict['TODOS']) # unidade em km
    # cluster de ocorrência de incidentes e relatos
    incidents_location('Localizacao_IncidentesRelatos_','TODOS',allIncidentsReportsDict['TODOS']) # unidade em km
    # intervalo em tempo de incidentes consecutivas
    interArrrival_time_distribution('Intervalo_Tempo_Incidentes_','TODOS',allIncidentsDict['TODOS'], nbins=24,limit = 2*3600) # unidade em segundos
    # intevalo em tempo de relatos consecutivos
    interArrrival_time_distribution('Intervalo_Tempo_Relatos_','TODOS',allReportsDict['TODOS'], nbins=24,limit = 2*3600) # unidade em segundos
    # intevalo em tempo de incidentes + relatos consecutivos
    interArrrival_time_distribution('Intervalo_Tempo_IncidentesRelatos_','TODOS',allIncidentsReportsDict['TODOS'], nbins=24,limit = 2*3600) # unidade em segundos
    # resumo TODOS
    plot_resume_cop("Resumo_TODOS.png",'TODOS',matchDays,actionsSerie['TODOS'],incidentsSerie['TODOS'],reportsSerie['TODOS'])
        
    #cops para os quais sao criados os graficos
    graphicsFromCops = ['CCDA - BHZ',
                        'CCDA - BSB',
                        'CCDA - FOR',
                        'CCDA - REC',
                        'CCDA - RIO',
                        'CCDA - SSA',
                        ]

    for cop in graphicsFromCops:
        # cluster de ocorrência de incidentes
        incidents_location('Localizacao_Incidentes_',cop,allIncidentsDict[cop]) # unidade em km
        # cluster de ocorrência de relatos
        incidents_location('Localizacao_Relatos_',cop,allReportsDict[cop]) # unidade em km
        # cluster de ocorrência de incidentes e relatos
        incidents_location('Localizacao_IncidentesRelatos_',cop,allIncidentsReportsDict[cop]) # unidade em km
        # intervalo em tempo de incidentes consecutivas
        interArrrival_time_distribution('Intervalo_Tempo_Incidentes_',cop,allIncidentsDict[cop], nbins=24,limit = 2*3600) # unidade em segundos
        # intervalo em tempo de relatos consecutivos
        interArrrival_time_distribution('Intervalo_Tempo_Relatos_', cop,allReportsDict[cop], nbins=24,limit = 2*3600) # unidade em segundos
        # intevalo em tempo de incidentes + relatos consecutivos
        interArrrival_time_distribution('Intervalo_Tempo_IncidentesRelatos_',cop,allIncidentsReportsDict[cop], nbins=24,limit = 2*3600) # unidade em segundos
        #criacao dos clusters
        clusters, itensClusterizados = computeCluster('Cluster3D_IncidentesRelatos_',cop,allIncidentsReportsDict[cop])
        #resumo de cops
        plot_resume_cop("Resumo_"+cop+".png",cop,matchDays,actionsSerie[cop],incidentsSerie[cop],reportsSerie[cop])
        
        #for c in range(0,len(itensClusterizados)):
        #    #só posso fazer a contagem de intervalos se exister mais de um incidente no cluster
        #    if(len(itensClusterizados[c])>1):
        #        interArrrival_distance_distribution(cop+str(c),itensClusterizados[c], limit = 50) # unidade em segundos    
    
    # Dados finais

    print '-' * 100
    print 'TODOS'
    print "Total de incidentes", len(allIncidentsDict['TODOS'])
    compute_statistics(incidentsSerie['TODOS'])
    print "Total de relatos", len(allReportsDict['TODOS'])
    compute_statistics(reportsSerie['TODOS'])
    print "Total de ações", len(allActionsDict['TODOS'])
    compute_statistics(actionsSerie['TODOS'])
    print '-' * 100
    

    for cop in graphicsFromCops:
        print '-' * 100
        print cop
        print "Total de incidentes", len(allIncidentsDict[cop])
        compute_statistics(incidentsSerie[cop])
        print "Total de relatos", len(allReportsDict[cop])
        compute_statistics(reportsSerie['TODOS'])
        print "Total de ações", len(allActionsDict[cop])
        compute_statistics(actionsSerie[cop])
            