# -*- coding: utf-8 -*- 
from math import radians, cos, sin, asin, sqrt
from scipy.misc import factorial
from datetime import datetime
from scipy.optimize import curve_fit
import os
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
from scipy.stats import lomax,expon,powerlaw,bayes_mvs,pareto,truncexpon
import scipy.stats
import numpy as np
import random
#Constantes
dateDistanceLimit = 43200 #(12 horas em segundos)
actionSize = 43200 #(12 horas em segundos)
punctualActionSize = 0 #(1 hora em segundos)
inicioAmostragem = datetime(2014,6,12,0,0,0)
terminoAmostragem = datetime(2014,7,13,23,59,59)
#inicioAmostragem = datetime(2013,6,15,0,0,0)
#terminoAmostragem = datetime(2013,6,30,23,59,59)
#COPs avaliados
allCopsCopaConf = ['CCDA - BHZ',
            'CCDA - BSB',
            'CCDA - FOR',
            'CCDA - REC',
            'CCDA - RIO',
            'CCDA - SSA',
            'CC2 - FTC - SSA',
            'CCTI - SSA']

allCopsCopaMundo = [
            'CCDA - MAO', 'FNC - MAO', 'FTP', 'FTC','CCTI',
            'CCDA - NAT',
            'CCDA - FOR',
            'CCDA - REC','GCL Maceio',
            'CCDA - BHZ',
            'CCDA - BSB','CCom_BPEB_CCDA_Bsb',
            'CCDA - SAO',
            'CCDA - RIO','FTC Centro Norte (1 BI Mtz)',
            'CCDA - SSA','CC2 - FTC - SSA','CCTI - SSA',
            'CCDA - CTB','5 BDA C BLD','15 Bda Inf Mec',
            'CCDA - POA','FT Centro Sul (29 BIB)',
            'CCDA - CGB'
            ]

matchDays = [datetime(2013,6,15,0,0,0),datetime(2013,6,16,0,0,0),datetime(2013,6,17,0,0,0),datetime(2013,6,19,0,0,0),datetime(2013,6,20,0,0,0),
                 datetime(2013,6,22,0,0,0),datetime(2013,6,23,0,0,0),datetime(2013,6,26,0,0,0),datetime(2013,6,27,0,0,0),datetime(2013,6,30,0,0,0)]
    
mdays = [#datetime(2013,6,10),datetime(2013,6,11),datetime(2013,6,12),datetime(2013,6,13),
        #datetime(2013,6,14),
        datetime(2013,6,15),datetime(2013,6,16),datetime(2013,6,17),datetime(2013,6,18),datetime(2013,6,19),
        datetime(2013,6,20),datetime(2013,6,21),datetime(2013,6,22),datetime(2013,6,23),datetime(2013,6,24),
        datetime(2013,6,25),datetime(2013,6,26),datetime(2013,6,27),datetime(2013,6,28),datetime(2013,6,29),datetime(2013,6,30)]
        #datetime(2013,7,1)]

copadays = [datetime(2014,6,12),datetime(2014,6,13),datetime(2014,6,14),datetime(2014,6,15),datetime(2014,6,16),
            datetime(2014,6,17),datetime(2014,6,18),datetime(2014,6,19),datetime(2014,6,20),datetime(2014,6,21),
            datetime(2014,6,22),datetime(2014,6,23),datetime(2014,6,24),datetime(2014,6,25),datetime(2014,6,26),
            datetime(2014,6,27),
            datetime(2014,6,28),datetime(2014,6,29),datetime(2014,6,30),datetime(2014,7,1),
            datetime(2014,7,2),datetime(2014,7,3),### sem jogo
            datetime(2014,7,4),
            datetime(2014,7,5),
            datetime(2014,7,6),datetime(2014,7,7), ## sem jogo
            datetime(2014,7,8),datetime(2014,7,9),
            datetime(2014,7,10),datetime(2014,7,11), ##sem jogo
            datetime(2014,7,12),datetime(2014,7,13),
            ]

matchDays = copadays


# globais
resultados = {}

#coeficientes da distribuição LOMAX
coefDistribuicaoLomax = {}

# correlacao
correlacao = {}

#coeficientes R2
coefR2Lomax = {}

#distribuicao real do intervalo entre chegadas
distRealInterArrival = {}
def changeCop(cop):

    maceio = u"GCL Maceió"
    ftCentroNorte = u"FT Centro Norte (1º BI Mtz)"
    bdaInfMec = u"15º BDA INF MEC"
    ftCentroSul = u"FT Centro Sul (29º BIB)"

    if(cop == 'FNC - MAO' or cop == 'FTP' or cop == 'FTC' or cop =='CCTI'):
        return 'CCDA - MAO'

    if cop.encode("utf-8") == maceio.encode("utf-8"):
        return 'CCDA - REC'

    if cop == 'CCom_BPEB_CCDA_Bsb':
        return 'CCDA - BSB'
    
    if cop.encode("utf-8") == ftCentroNorte.encode("utf-8"):
        return 'CCDA - RIO'

    if cop == 'CC2 - FTC - SSA' or cop == 'CCTI - SSA':
        return 'CCDA - SSA'

    if cop == '5 BDA C BLD' or cop.encode("utf-8") == bdaInfMec.encode("utf-8"):
        return 'CCDA - CTB'

    if cop.encode("utf-8") == ftCentroSul.encode("utf-8"):
        return 'CCDA - POA'

    return cop

def get_available_cops():
    """
        Retorna todos os cops existentes no intervalo de amostragem
    """
    allIncidents = Incident.get_all()
    cops = []
    
    for i in allIncidents:
        if(inicioAmostragem <= i.reporting_date and i.reporting_date <=terminoAmostragem):
            cops.append(i['operations_center']['id'])
            #cops.append(i['operations_center'])
                
    allReports = RelatoDeSituacao.get_all()
    
    for r in allReports:
        if (
                inicioAmostragem <= r.data_hora and 
                r.data_hora <=terminoAmostragem and
                'cop' in r.relator 
                 and # todos tem que ter o COP
                 'id' in r.relator['cop']  # todos tem que ter o id               
            ):
                cops.append(r.relator['cop']['id'])
                #cops.append(r.relator['cop'])
    
    return set(cops)


def get_dict_all_actions():
    """
        Retorna todas as ações agrupadas em um dicionário cuja chave é o nome do COP
    """
    allSincronizations = Sincronizacao.get_all()
    dictionaryAllActions = {}
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
            (i['operations_center']['id'] in allCops) and
            #(i['operations_center'] in allCops) and
            (inicioAmostragem <= i.reporting_date and i.reporting_date <=terminoAmostragem)
        ):
        
            i['operations_center']['id'] = changeCop(i['operations_center']['id'])
            #i['operations_center'] = changeCop(i['operations_center'])
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
        dictionaryAllIncidents[incident['operations_center']['id']].append(incident)
        #dictionaryAllIncidents[incident['operations_center']].append(incident)
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
    
    reports = []
    for r in allReports:
        if (
                inicioAmostragem <= r.data_hora and 
                r.data_hora <=terminoAmostragem and
                'cop' in r.relator 
                and # todos tem que ter o COP
                'id' in r.relator['cop'] and # todos tem que ter o COP
                r.relator['cop']['id'] in allCops
            ):
                r.relator['cop']['id'] = changeCop(r.relator['cop']['id'])
                #r.relator['cop'] = changeCop(r.relator['cop'])
                reports.append(r)
    return reports
        
def get_dict_all_reports():
    """
        Retorna todos os relatos agrupados em um dicionário cuja chave é o nome do COP
    """
    dictionaryAllReports = {}
        
    for cop in allCops:
        dictionaryAllReports[cop] = []
    dictionaryAllReports['TODOS'] = []

    allReports = get_all_reports()
    for report in allReports:
        dictionaryAllReports['TODOS'].append(report)
        dictionaryAllReports[report.relator['cop']['id']].append(report)
        #dictionaryAllReports[report.relator['cop']].append(report)
                
    return dictionaryAllReports

def get_reports_near_date(listReports,date,mask = '%Y/%m/%d'):
    """
    Retorna todas os relatos de situação de "listReports" nas quais o valor "date" seja igual, segundo "MASK", ao data_hora
    Por padrão, MASK considera apenas ANO/MES/DIA, com hora 00:00:00
    """
    return [report for report in listReports
            if datetime.strptime(datetime.strftime(report.data_hora,mask),mask) == datetime.strptime(datetime.strftime(date,mask),mask)
    ]

def dateChangeFormat(item):

    return datetime.strftime(item,"%d/%m")


def plot_interArrival(samples,labels,padroes,filename,title):

    axisX = range(0,len(samples[0]))
    plt.close('all')
    fig = plt.figure()
    fig.suptitle(cop+"\n"+title)
    for sample,label,padrao in zip(samples,labels,padroes):
        plt.plot(axisX,sample,padrao,label=label)
    plt.ylabel("P(X<=x)")
    plt.xlabel("Interval [minutes]")
    plt.xticks(axisX,rotation=45)
    plt.grid(True)
    fig.set_size_inches(18.5,10.5)
    plt.legend(prop={'size':12},bbox_to_anchor=(0.99, 0.4))

    fig.savefig(filename,dpi=96)
    plt.close('all')

def plot_resume_cop(filename,cop,axisX,actions,incidents,reports):

    plt.close('all')
    fig = plt.figure()
    """
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
    """
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

    graphIncRelsActions = plt.subplot2grid((1,1),(0,0))      
    graphIncRelsActions.set_title(cop+"\nPearson Correlation: " + str(stats.pearsonr(actions,incRel)[0]))
    graphIncRelsActions.set_ylabel("Quantity [Units]")
    graphIncRelsActions.set_xlabel("Days")
    linesIncRelsActions = graphIncRelsActions.plot(axisX,incRel, 'g^-',axisX,actions, 'bo-')
    graphIncRelsActions.xaxis_date()
    graphIncRelsActions.xaxis.set_major_formatter(DateFormatter("%B,%d"))
    plt.xticks(axisX,rotation=45)
    graphIncRelsActions.grid(True)
    plt.legend(iter(linesIncRelsActions),('Information','Actions'),prop={'size':12},bbox_to_anchor=(1, 1))
    plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
    fig.set_size_inches(18.5,10.5)
    if(os.path.exists==False):
        os.mkdir(cop)   
    fig.savefig(cop+'/'+filename,dpi=96)
    correlacao[cop]=stats.pearsonr(actions,incRel)[0]
    #correlacao.append(stats.pearsonr(actions,incRel)[0])
 
def funcExponential(x,a):

    return  1.0 - (np.exp(-a*x))
    #return lomax.cdf(x,a)

def pdfExponential(x,a):

    return  a * (np.exp(-a*x))
    #return lomax.pdf(x,a)

def pdfLomax(x,a,b):

    return  (a * np.power(b,a)) / np.power(x+b,a+1)
    #return  (a * np.power(b,a)) / np.power(x,a+1)
    

def funcLomax(x,a,b):
    
    return 1 - (np.power(b,a)/(np.power(x+b,a)))
    #return 1 - (np.power(b,a)/(np.power(x,a)))
    #return lomax.cdf(x,a)
    #return genpareto.cdf(x,a)

def funcLomaxPonderada(x):
    
    #a = 3.86607243476
    #b = 90.3290141236
    a = 1.42421085234
    b = 10.3310560189
    #a = 1.99249027486
    #b = 14.7003560784
    return 1 - (np.power(b,a)/(np.power(x+b,a)))
    #return 1 - (np.power(b,a)/(np.power(x,a)))
    #return lomax.cdf(x,a)
    #return genpareto.cdf(x,a)

def funcLomaxAritmetica(x):
    
    #a = 3.86607243476
    #b = 90.3290141236
    #a = 1.42421085234
    #b = 10.3310560189
    a = 1.99249027486
    b = 14.7003560784
    return 1 - (np.power(b,a)/(np.power(x+b,a)))
   
def compute_statistics(serie):
    """
        Computa as estatísticas de SERIE utilizando stats.describe
    """
    sizeData, (minimum,maximum),arithmeticMean,variance,skeness,kurtosis = stats.describe(serie)

    print "Size Data  = ",sizeData , "Minimo,Maximo = ",(minimum,maximum), "Média = ", arithmeticMean , "Variância = ", variance

def info_distribution(filename,cop,serie, nbins=30,limit = 24*3600,cor='green'):

    ocorrencias = [0] * 24
       
    for i in serie:
        if (hasattr(i,'reporting_date')): # é incidente         
            ocorrencias[int(datetime.strftime(i.reporting_date,"%H"))] = ocorrencias[int(datetime.strftime(i.reporting_date,"%H"))] + 1    

        elif (hasattr(i,'data_hora')): # é relato
            ocorrencias[int(datetime.strftime(i.data_hora,"%H"))] = ocorrencias[int(datetime.strftime(i.data_hora,"%H"))] + 1
    
    # Criando grafico com distribuico de incidentes+relatos pelas horas do dia

    qtde, bins, patches = plt.hist(ocorrencias, 24,facecolor='r', alpha=0.5)
   
    plt.close('all')
    fig = plt.figure()
    plt.bar(range(0,24),ocorrencias, align='center')
    fig.suptitle(cop+"\nInformation Distribution")
    plt.ylabel("Quantity [Units]")
    plt.xlabel("Hour")
    plt.xticks(range(0,24),rotation=45)
    plt.grid(True)
    fig.set_size_inches(18.5,10.5)
    if(os.path.exists==False):
        os.mkdir(cop)   
    fig.savefig(cop+'/'+'HORA_'+cop+'.png',dpi=96)
    plt.close('all')

def testandoDistribuicao(sample):

    cdfs =[
            "expon",           #Exponential
            "genexpon",        #Generalized Exponential
            "genpareto",       #Generalized powerlaw
            "lomax"           #Lomax
            
        ]
    for cdf in cdfs:
        #fit our data set against every probability distribution
        parameters = eval("scipy.stats."+cdf+".fit(sample)");
     
        #Applying the Kolmogorov-Smirnof one sided test
        D, p = scipy.stats.kstest(sample, cdf, args=parameters);
        #pretty-print the results
        print cdf.ljust(16) + ("p: "+str(p)).ljust(25)+"D: "+str(D)+ "parametros = ",str(parameters).ljust(40)

def interArrrival_time_distribution(cop,serie, nbins=30,limit = 24*3600,cor='green'):

    """
        Calcula a distribuição dos tempos entre ocorrencias dos incidentes.
        Salva em arquivo
    """
    if(os.path.exists(cop)==False):
        os.mkdir(cop)   

    arrivalTime = []
        
    ocorrencias = [0] * 24
       
    for i in serie:
        if (hasattr(i,'reporting_date')): # é incidente
            arrivalTime.append(datetime.strptime(datetime.strftime(i.reporting_date,"%Y-%m-%d %H:%M:%S"),"%Y-%m-%d %H:%M:%S"))

        elif (hasattr(i,'data_hora')): # é relato
            arrivalTime.append(datetime.strptime(datetime.strftime(i.data_hora,"%Y-%m-%d %H:%M:%S"),"%Y-%m-%d %H:%M:%S"))
        
    sortedArrivalTime =  sorted(arrivalTime)

    interArrivalTime = []
    if(len(sortedArrivalTime)>0):
        for i in range(0,len(sortedArrivalTime)-1):
            if(sortedArrivalTime[i+1]!=sortedArrivalTime[i]):
                interArrivalTime.append(((sortedArrivalTime[i+1] - sortedArrivalTime[i]).total_seconds())/60.0)
            
    percentagemInterArrivalTime = []
    cdfQtdeInterArrivalTime = []
    qtdeInterArrivalTime = []
    qtdeAbsoluta = []
    axisX = []

    
    for t in np.arange(0,61,1):
        # a qtde eh armazenada como float por causa de divisao ... para resultar em float
        #percentagemInterArrivalTime.append(float(len([q for q in interArrivalTime if (q <= t)])))
        cdfQtdeInterArrivalTime.append(float(len([q for q in interArrivalTime if (q <= t)])))
        qtdeInterArrivalTime.append(float(len([q for q in interArrivalTime if (t < q < t+1)])))
        qtdeAbsoluta.append(float(len([q for q in interArrivalTime if (q <= t)])))
        axisX.append(t)

    if(len(interArrivalTime)>0):
        
        # pego só o último que representa o maior valor do intervalo. Existem valores maiores que 60 minutos que são desconsiderados
        total = qtdeAbsoluta[-1]

        percentagemInterArrivalTime = [q/float(total) for q in qtdeAbsoluta]

        #armazena distribuicao real dos intervalos entre chegadas no dict
        distRealInterArrival[cop]=percentagemInterArrivalTime

        plt.close('all')
        fig = plt.figure()                
        #poptExp, pocvExp = curve_fit(funcExponential,np.array(axisX),percentagemInterArrivalTime,maxfev=2000)
        poptLomax, pocvLomax = curve_fit(funcLomax,np.array(axisX),percentagemInterArrivalTime,maxfev=3000)       

        #armazena os coeficientes no dict
        coefDistribuicaoLomax[cop]=poptLomax

        #expoR2 = computeR2(percentagemInterArrivalTime,funcExponential(np.array(axisX),*poptExp))
        lomaxR2 = computeR2(percentagemInterArrivalTime,funcLomax(np.array(axisX),*poptLomax))

        #coeficientes R2 no dict
        coefR2Lomax[cop]=lomaxR2
        

        seriesPlotted = plt.plot(
            axisX,funcLomax(np.array(axisX),*poptLomax),'g*-',
            axisX,percentagemInterArrivalTime,'ro-',
        )

        fig.suptitle(cop+"\nInter-arrival - Ajuste Lomax X Real")
        plt.ylabel("P(X<=x)")
        plt.xlabel("Interval [minutes]")
        plt.xticks(axisX,rotation=45)
        plt.grid(True)
        fig.set_size_inches(18.5,10.5)
        plt.legend(iter(seriesPlotted),('Ajuste Lomax','Real'),prop={'size':12},bbox_to_anchor=(1, 0.1))
        #fig.savefig(cop+'/'+'cdf_'+filename+cop+'.png',dpi=96)
        fig.savefig('CDF_ajusteLomax_X_Real/'+cop+'.png',dpi=96)
        #fig.savefig('porcentagem/'+filename+cop+'.png',dpi=96)
        plt.close('all')
        
        # parte do historico de interarrival = pdf mais ou menos
        tmpQtde = np.sum(qtdeInterArrivalTime)
        qtdeInterArrivalTime = [float(q)/tmpQtde for q in qtdeInterArrivalTime]

        plt.close('all')
        fig = plt.figure()
        fig.suptitle(cop+"\nInter-arrival time")
        plt.plot(
            axisX,qtdeInterArrivalTime,'ro-', 
        )
        plt.ylabel("Quantity [Units]")
        plt.xlabel("Interval [minutes]")
        plt.xticks(axisX,rotation=45)
        plt.grid(True)
        fig.set_size_inches(18.5,10.5)
        #plt.legend(iter(seriesPlotted),('Exponential'),prop={'size':12},bbox_to_anchor=(1, 0.1))
        #fig.savefig(cop+'/'+'quantity_'+filename+cop+'.png',dpi=96)
        fig.savefig('PDF_Real/'+cop+'.png',dpi=96)
        plt.close('all')
        
def computeR2(y, fy):

    #y = percentagemQtde 
    #fy = funcGenPareto(axisX, *poptPareto))
    
    """
    ss_res = np.dot((y - fy),(y - fy))
    ymean = np.mean(y)
    ss_tot = np.dot((y-ymean),(y-ymean))
    return 1-ss_res/ss_tot #coeficiente R2
    """
    ybarra = np.sum(y)/float(len(y))
    numerador = float(np.sum([np.power(yi - fyi,2) for yi,fyi in zip(y,fy)]))
    denominador = float(np.sum([np.power(yi - ybarra,2) for yi in y]))
    return 1 - numerador/denominador

def erroMedio(real,ajuste):

    # calcula a media da diferenca, em modulo, entre y e ajuste
    erro = 0
    arrayErro=[]
    for reali, ajustei in zip(real,ajuste):

        if(reali == 0):
            arrayErro.append(abs(reali-ajustei))
            #erro = erro + (abs(reali-ajustei))
        else:
            arrayErro.append(abs(reali-ajustei)/reali)
            #erro = erro + (abs(reali-ajustei)/reali)
    #return erro/float(len(real))
    return compute_statistics(arrayErro)

    
if __name__ == "__main__":
    """
        Loop principal
    """
    
    allCops = get_available_cops()

    allActionsDict = get_dict_all_actions()
    allIncidentsDict = get_dict_all_incidents()
    allReportsDict = get_dict_all_reports()
    allIncidentsReportsDict = {}

    incidentsSerie = {}
    actionsSerie = {}
    reportsSerie = {}
    incidentsReportsSerie = {}

    incidentsSerie['TODOS'] = []
    actionsSerie['TODOS'] = []
  
    reportsSerie['TODOS'] = []
    incidentsReportsSerie['TODOS'] = []
    for day in matchDays:
            incidentsSerie['TODOS'].append(len(get_incidents_near_date(allIncidentsDict['TODOS'],day)))
            actionsSerie['TODOS'].append(len(get_actions_near_date(allActionsDict['TODOS'],day)))
            reportsSerie['TODOS'].append(len(get_reports_near_date(allReportsDict['TODOS'],day)))
    
    # agrupar incidentes e relatos
    incidentsReportsSerie['TODOS'] = [i + r for i,r in zip(incidentsSerie['TODOS'],reportsSerie['TODOS'])]
    allIncidentsReportsDict['TODOS'] = allIncidentsDict['TODOS'] + allReportsDict['TODOS']
    
    for cop in allCops:
        incidentsSerie[cop]=[]
        actionsSerie[cop]=[]
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

    interArrrival_time_distribution('TODOS',allIncidentsReportsDict['TODOS'], nbins=60,limit =  1 * 3600) # unidade em segundos
    #info_distribution('Distribuicao de Info por horas','TODOS',allIncidentsReportsDict['TODOS'], nbins=24,limit = 24*3600,cor='green')
    plot_resume_cop("Resumo_TODOS.png",'TODOS',matchDays,actionsSerie['TODOS'],incidentsSerie['TODOS'],reportsSerie['TODOS'])
    
    #cops para os quais sao criados os graficos  
    graphicsFromCops = [
                        'CCDA - RIO',
                        'CCDA - MAO', 
                        'CCDA - NAT',
                        'CCDA - FOR',
                        'CCDA - REC',
                        'CCDA - BHZ',
                        #'CCDA - BSB',
                        'CCDA - SAO',
                        'CCDA - SSA',
                        'CCDA - CTB',
                        'CCDA - POA',
                        'CCDA - CGB',
                        #'TODOS'
                        ]

    for cop in graphicsFromCops:
        interArrrival_time_distribution(cop,allIncidentsReportsDict[cop], nbins=60,limit = 1 * 3600) # unidade em segundos
        #info_distribution('Distribuicao de Info por horas',cop,allIncidentsReportsDict[cop], nbins=24,limit = 24*3600,cor='green')
        plot_resume_cop("Resumo_"+cop+".png",cop,matchDays,actionsSerie[cop],incidentsSerie[cop],reportsSerie[cop])

    # Dados finais  
    
    # media ponderada das probabilidades. Correlacao x P(X<=x)
    arrayDistRealMediaPonderada = []

    for t in range(0,61):
        probT = 0
        for cop in graphicsFromCops:
            probT = probT + correlacao[cop]*distRealInterArrival[cop][t]
        arrayDistRealMediaPonderada.append(probT/float(np.sum(correlacao.values())))

    # ajustando a distribuicao media ponderada
    poptRealMediaPonderada, pocvRealMediaPonderada = curve_fit(funcLomax,range(0,61),arrayDistRealMediaPonderada,maxfev=3000)

    arrayDistLomaxMediaPonderada = funcLomax(range(0,61),poptRealMediaPonderada[0],poptRealMediaPonderada[1])
    print 'Parametros da Lomax para dist real podenrada = ', poptRealMediaPonderada
    print 'ajuste = ', computeR2(arrayDistRealMediaPonderada,arrayDistLomaxMediaPonderada)
    
    #plotando comparacao das funcoes da distribuicao reais
    plot_interArrival([arrayDistRealMediaPonderada,arrayDistLomaxMediaPonderada],['Real','Ponderada'],['ro-','gx-'],'realPonderada_LomaxPonderada.png','Comparacao entre Distribuicao Real e Ajustada')
    #plot_interArrival([distRealInterArrival['TODOS'],arrayDistRealMediaPonderada],['Real','Ponderada'],['ro-','gx-'],'TODOS'+'_Real_RealPonderada.png','Comparacao Distribuicao Real de TODOS e Distribuicao Real Geral Ponderada')
    #plot_interArrival([funcLomax(range(0,61),coefDistribuicaoLomax['TODOS'][0],coefDistribuicaoLomax['TODOS'][1]),arrayDistLomaxMediaPonderada],['Real','Ponderada'],['ro-','gx-'],'TODOS'+'_Lomax_LomaxPonderada.png','Comparacao Distribuicao Ajustada de TODOS e Distribuicao Lomax Geral Ponderada')

    #plotando comparacao de cada COP com a distribuicao ponderada
    #for cop in graphicsFromCops:
    #    plot_interArrival([distRealInterArrival[cop],arrayDistRealMediaPonderada],['Real','Real Geral Ponderada'],['ro-','gx-'],cop+'/Real_RealPonderada.png','Comparacao Distribuicao Real de '+ cop +' e Distribuicao Real Geral Ponderada')
    #    plot_interArrival([funcLomax(range(0,61),coefDistribuicaoLomax[cop][0],coefDistribuicaoLomax[cop][1]),arrayDistLomaxMediaPonderada],['Real','Lomax Geral Ponderada'],['ro-','gx-'],cop+'/Lomax_LomaxPonderada.png','Comparacao Distribuicao Ajustada de '+ cop + ' e Distribuicao Lomax Geral Ponderada')
    #    plot_interArrival([distRealInterArrival[cop],arrayDistLomaxMediaPonderada],['Real','Lomax Geral Ponderada'],['ro-','gx-'],cop+'/Real_LomaxPonderada.png','Comparacao Distribuicao Real de '+ cop +' e Distribuicao Lomax Geral Ponderada')
        #------tabela
    #    print cop,'|', correlacao[cop],'|',coefR2Lomax[cop],'|',coefDistribuicaoLomax[cop],'|',computeR2(distRealInterArrival[cop],arrayDistRealMediaPonderada),'|',computeR2(funcLomax(range(0,61),coefDistribuicaoLomax[cop][0],coefDistribuicaoLomax[cop][1]),arrayDistLomaxMediaPonderada),'|', computeR2(distRealInterArrival[cop],arrayDistLomaxMediaPonderada)

    """
    arrayDist=[]
    for cop in graphicsFromCops:
        arrayDist.append(distRealInterArrival[cop])  
    plot_interArrival(arrayDist,graphicsFromCops,['bo-','ko-','mo-','yo-','go-','ro-','bx-','kx-','mx-','yx-','gx-'],'todasReal.png','Comparacao entre todas Distribuicoes Reais')
 
    # gerando um grafico unico com todas as distribuicoes ajustadas para lomax
    arrayDist=[]
    for cop in graphicsFromCops:
        arrayDist.append(funcLomax(range(0,61),coefDistribuicaoLomax[cop][0],coefDistribuicaoLomax[cop][1]))
    plot_interArrival(arrayDist,graphicsFromCops,['bo-','ko-','mo-','yo-','go-','ro-','bx-','kx-','mx-','yx-','gx-'],'todasLomax.png','Comparacao entre todas Distribuicoes Ajustadas')
    """

    # calculo das constantes alpha e beta de ajuste

    alfa = {}
    beta = {}
    combinacaoAlfaBeta = {}
    combinacaoR2_Superior = {}
    distCombinacao = {}
    print 'TODOS Erro medio - Ajuste Proprio = ', erroMedio(distRealInterArrival['TODOS'],funcLomax(range(0,61),coefDistribuicaoLomax['TODOS'][0],coefDistribuicaoLomax['TODOS'][1]))
    for cop in graphicsFromCops:
        print 'Gerando estudo de ', cop
        alfa[cop]=[]
        beta[cop]=[]
        combinacaoAlfaBeta[cop] = []
        combinacaoR2_Superior[cop] = []
        distCombinacao[cop] = []
        """"
        ---> tentativa #1
        for f,r in zip(arrayDistLomaxMediaPonderada,distRealInterArrival[cop]):
            
            if(f==0):
                alfa[cop].append(0.0)
                beta[cop].append(0.0)
            else:
                alfa[cop].append((2*r/f)-1)
                beta[cop].append(-r + f)
            

        #testando melhor combinacao de alfa e beta. A melhor é aquela que produz o maior R2
        #distCombinacao=[]
        for a in alfa[cop][1:]:
            for b in beta[cop][1:]:
                distCombinacao[cop].append([(a*f + b) for f in arrayDistLomaxMediaPonderada])
                combinacaoAlfaBeta[cop].append([a,b])
                combinacaoR2_Superior[cop].append(computeR2(distCombinacao[cop],arrayDistLomaxMediaPonderada))
        posMelhorCombinacao = combinacaoR2_Superior[cop].index(np.amax(combinacaoR2_Superior[cop]))
        #print 'melhor r2 de alfa beta = ',np.amax(combinacaoR2_Superior[cop])
        print 'melhor r2 de alfa beta = ',combinacaoR2_Superior[cop][posMelhorCombinacao]
        print 'alfa e beta = ', combinacaoAlfaBeta[cop][posMelhorCombinacao]
        plot_interArrival([distRealInterArrival[cop],arrayDistLomaxMediaPonderada,distCombinacao[cop][posMelhorCombinacao]],['Real','Lomax Media','AlfaBeta'],['ro-','mo-','gx-'],cop+'_alfaBeta.png','Alfa Beta e a Real')

        """
        # tentativa #2
        print 'Erro medio - Ajuste proprio= ', erroMedio(distRealInterArrival[cop],funcLomax(range(0,61),coefDistribuicaoLomax[cop][0],coefDistribuicaoLomax[cop][1]))
        #print 'Erro medio - Ajuste Media Geral Ponderada= ', erroMedio(distRealInterArrival[cop],arrayDistLomaxMediaPonderada)
        #plot_interArrival([alfa[cop],beta[cop]],['alfa','beta'],['bo-','go-'],cop+'_coefAlphaBeta.png','Coef alfa e beta')
    print 'Real Ponderada - Erro Medio - Ajuste proprio', erroMedio(arrayDistRealMediaPonderada,arrayDistLomaxMediaPonderada)