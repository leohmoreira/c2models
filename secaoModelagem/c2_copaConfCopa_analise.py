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
from scipy.stats import lomax,expon,powerlaw,bayes_mvs,pareto,truncexpon,norm
import scipy.stats
import numpy as np
import random
#Constantes
dateDistanceLimit = 43200 #(12 horas em segundos)
actionSize = 43200 #(12 horas em segundos)
punctualActionSize = 0 #(1 hora em segundos)
#Copa do mundo
inicioAmostragem = datetime(2014,6,12,0,0,0)
terminoAmostragem = datetime(2014,7,13,23,59,59)
#Copa das Confederaçoes
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
            'CCDA - BSB','CCom_BPEB_CCDA_Bsb','CMP','CCom_32 GAC_FTC_CCDA_Bsb','CCom_3 Esqd C Mec_FTC_CCDA_Bsb','CCom_41BIMtz_FTC_CCDA_Bsb',
            'CCom_41BIMtz_FTC_CCDA_Bsb','CCom_16 B Log_FTC_CCDA_Bsb','32GAC','CCom_22BI_FTC_CCDA_Bsb','CCom_BGP_FTC_CCDA_Bsb', 'CCom_36BIMtz_FTC_CCDA_Bsb',
            'CCDA - SAO',
            'CCDA - RIO','FTC Centro Norte (1 BI Mtz)',
            'CCDA - SSA','CC2 - FTC - SSA','CCTI - SSA',
            'CCDA - CTB','5 BDA C BLD','15 Bda Inf Mec','FTC CTB'
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
#conf matchDays = mdays


# globais
resultados = {}

#coeficientes da distribuição LOMAX
coefDistribuicaoLomax = {}

#coeficientes da distribuição Expo
coefDistribuicaoExpo = {}

#coeficientes da distribuição Weibull
coefDistribuicaoWeibull = {}

# correlacao
correlacao = {}

#coeficientes R2
coefR2Lomax = {}
coefR2LomaxI = {}
coefR2Expo = {}
coefR2Weibull = {}


#distribuicao real do intervalo entre chegadas = CDF
distRealInterArrival = {}

#distribuicao real do intervalo entre chegadas = PDF
distRealPDF = {}

#data limite para cada COP (maior data de incidente, relato ou ação). Utilizado para calculo da correlacao
greatestDate = {}

# qtde de eventos por COp
qtdeEventos = {}

#eixoX de cada COP
axisXCop = {}
def changeCop(cop):

    maceio = u"GCL Maceió"
    ftCentroNorte = u"FT Centro Norte (1º BI Mtz)"
    bdaInfMec = u"15º BDA INF MEC"
    ftCentroSul = u"FT Centro Sul (29º BIB)"
    ccom32GAC = u"CCom_32º GAC_FTC_CCDA_Bsb"
    ccom3Esqd= u"CCom_3º Esqd C Mec_FTC_CCDA_Bsb"
    ccom41BIMTZ= u"CCom_41BIMtz_FTC_CCDA_Bsb"
    ccom16Blog= u"CCom_16º B Log_FTC_CCDA_Bsb"
    gac32=u"32°GAC"
    ccom22BI=u"CCom_22BI_FTC_CCDA_Bsb"
    ccomBGP=u"CCom_BGP_FTC_CCDA_Bsb"
    ccom36BI=u"CCom_36BIMtz_FTC_CCDA_Bsb"
    ccomRCG=u"CCom_1º RCGd_CCDA_Bsb"

    if(cop in ['FNC - MAO','FTP','FTC','CCTI','FOCON']):
        return 'CCDA - MAO'

    if cop =='FNC':
        return 'CCDA - NAT'

    if cop.encode("utf-8") == maceio.encode("utf-8"):
        return 'CCDA - REC'

    if ((cop in ['COC','FNC_MB','CCom_BPEB_CCDA_Bsb','CMP','CCom_41BIMtz_FTC_CCDA_Bsb','CCom_22BI_FTC_CCDA_Bsb','CCom_BGP_FTC_CCDA_Bsb','CCom_36BIMtz_FTC_CCDA_Bsb']) or (cop.encode("utf-8") == ccom32GAC.encode("utf-8") or 
        cop.encode("utf-8") == ccom3Esqd.encode("utf-8") or
        cop.encode("utf-8") == ccom16Blog.encode("utf-8") or
        cop.encode("utf-8") == gac32.encode("utf-8") or
        cop.encode("utf-8") == ccomRCG.encode("utf-8")
        )):
        return 'CCDA - BSB'
    
    if cop.encode("utf-8") == ftCentroNorte.encode("utf-8"):
        return 'CCDA - RIO'

    if cop == 'CC2 - FTC - SSA' or cop == 'CCTI - SSA':
        return 'CCDA - SSA'

    if cop in ['5 BDA C BLD','FTC CTB'] or cop.encode("utf-8") == bdaInfMec.encode("utf-8"):
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
#conf            cops.append(i['operations_center'])
                
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
#conf                cops.append(r.relator['cop'])

    allSincronizations = Sincronizacao.get_all()
    for sinc in allSincronizations:
        for action in sinc.acoes:
            if (              
                    ((action.tipo == 'PONTUAL') and (action.inicio >= inicioAmostragem) and (action.inicio <= terminoAmostragem)) or
                    ((action.tipo == 'INTERVALO') and (action.inicio >= inicioAmostragem and action.fim <= terminoAmostragem))
            ):
                cops.append(sinc.cop_responsavel['id'])
    print set(cops)
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
                sinc.cop_responsavel['id'] = changeCop(sinc.cop_responsavel['id'])
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
                sinc.cop_responsavel['id'] = changeCop(sinc.cop_responsavel['id'])
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

def get_actions_greatest_date(listActions,mask = '%Y/%m/%d'):
    if(len(listActions)>0):
        datas = [act.inicio for act in listActions]
        return datetime.strptime(datetime.strftime(np.amax(datas),mask),mask)
    else:
        return inicioAmostragem

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
#conf            (i['operations_center'] in allCops) and
            (inicioAmostragem <= i.reporting_date and i.reporting_date <=terminoAmostragem)
        ):
        
            i['operations_center']['id'] = changeCop(i['operations_center']['id'])
#conf            i['operations_center'] = changeCop(i['operations_center'])
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
#conf        dictionaryAllIncidents[incident['operations_center']].append(incident)
    return dictionaryAllIncidents


def get_incidents_near_date(listIncidents,date,mask = '%Y/%m/%d'):
    """
    Retorna todas os incidentes de "listIncidentes" nas quais o valor "date" seja igual, segundo "MASK", ao reporting_date
    Por padrão, MASK considera apenas ANO/MES/DIA, com hora 00:00:00
    """
    return [incident for incident in listIncidents
            if datetime.strptime(datetime.strftime(incident.reporting_date,mask),mask) == datetime.strptime(datetime.strftime(date,mask),mask)
    ]

def get_incidents_greatest_date(listIncidents, mask = '%Y/%m/%d'):
   
    if(len(listIncidents)>0):
        datas = [inc.reporting_date for inc in listIncidents]
        return datetime.strptime(datetime.strftime(np.amax(datas),mask),mask)
    else:
        return inicioAmostragem

def get_all_reports():
    """
        Retorna todos os relatos de situação agrupados em um array
    """
    allReports = RelatoDeSituacao.get_all()
    print 'TAMANHo = ', len(allReports)
    reports = []
    for r in allReports:
        if (
                inicioAmostragem <= r.data_hora and 
                r.data_hora <=terminoAmostragem and
                'cop' in r.relator 
                and # todos tem que ter o COP
                'id' in r.relator['cop'] and # todos tem que ter o COP
                r.relator['cop']['id'] in allCops
#conf                r.relator['cop'] in allCops
            ):
                r.relator['cop']['id'] = changeCop(r.relator['cop']['id'])
#conf                r.relator['cop'] = changeCop(r.relator['cop'])
                reports.append(r)
    print len(reports)
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
#conf        dictionaryAllReports[report.relator['cop']].append(report)
                
    return dictionaryAllReports

def get_reports_near_date(listReports,date,mask = '%Y/%m/%d'):
    """
    Retorna todas os relatos de situação de "listReports" nas quais o valor "date" seja igual, segundo "MASK", ao data_hora
    Por padrão, MASK considera apenas ANO/MES/DIA, com hora 00:00:00
    """
    return [report for report in listReports
            if datetime.strptime(datetime.strftime(report.data_hora,mask),mask) == datetime.strptime(datetime.strftime(date,mask),mask)
    ]

def get_reports_greatest_date(listReports, mask = '%Y/%m/%d'):
    if(len(listReports)>0):
        datas = [rep.data_hora for rep in listReports]
        return datetime.strptime(datetime.strftime(np.amax(datas),mask),mask)
    else:
        return inicioAmostragem

def dateChangeFormat(item):

    return datetime.strftime(item,"%d/%m")


def plot_interArrival(samples,labels,padroes,filename,title,limite=-1):

    axisX = range(0,len(samples[0]))
    plt.close('all')
    fig = plt.figure()
    #fig.suptitle(cop+"\n"+title)
    fig.suptitle(title)
    for sample,label,padrao in zip(samples,labels,padroes):
        plt.plot(axisX[0:limite],sample[0:limite],padrao,label=label,lw=3.0,ms=10.0)
    plt.ylabel("P(X <= t)")
    #plt.ylabel("Quantity [units]")
    #plt.ylabel("P(t < X < t + 1)")
    plt.xlabel("Interval [minutes]")
    plt.xticks(axisX[0:limite],rotation=45)
    plt.grid(True)
    fig.set_size_inches(18.5,10.5)
    plt.legend(prop={'size':16},bbox_to_anchor=(0.99, 0.5))

    fig.savefig(filename,dpi=96)
    plt.close('all')

def plot_resume_cop(filename,cop,axisX,actions,incidents,reports):

    plt.close('all')
    fig = plt.figure()
    # correlacao entre acoes e (incidentes + relatos)

    incRel = [i+r for i,r in zip(incidents,reports)]

    graphIncRelsActions = plt.subplot2grid((1,1),(0,0))      
    graphIncRelsActions.set_title(cop+"\nPearson Correlation: " + str(stats.pearsonr(actions,incRel)[0]))
    graphIncRelsActions.set_ylabel("Quantity [Units]")
    graphIncRelsActions.set_xlabel("Days")
    linesIncRelsActions = graphIncRelsActions.plot(axisX,incRel, 'g^-',axisX,actions, 'bo-',lw=3.0,ms=10.0)
    graphIncRelsActions.xaxis_date()
    graphIncRelsActions.xaxis.set_major_formatter(DateFormatter("%B,%d"))
    plt.xticks(axisX,rotation=45)
    graphIncRelsActions.grid(True)
    plt.legend(iter(linesIncRelsActions),('Observations','Actions'),prop={'size':16},bbox_to_anchor=(1, 1))
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

def funcLomaxI(x,a):
    
    return 1 - (1/(np.power(x+1,a)))

def funcWeibull(x,a,b):

    return 1 - (np.exp(-(np.power((x/a),b))))

   
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
        print 'PARAMETROS = ', parameters
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
        poptExp, pocvExp = curve_fit(funcExponential,np.array(axisX),percentagemInterArrivalTime,maxfev=3000)
        poptLomax, pocvLomax = curve_fit(funcLomax,np.array(axisX),percentagemInterArrivalTime,maxfev=3000)
        poptLomaxI, pocvLomaxI = curve_fit(funcLomaxI,np.array(axisX),percentagemInterArrivalTime,maxfev=3000)
        poptWeibull, pocvWeibull = curve_fit(funcWeibull,np.array(axisX),percentagemInterArrivalTime,maxfev=5000)

        #armazena os coeficientes no dict
        coefDistribuicaoLomax[cop]=poptLomax
        coefDistribuicaoExpo[cop]=poptExp
        coefDistribuicaoWeibull[cop]=poptWeibull

        expoR2 = computeR2(percentagemInterArrivalTime,funcExponential(np.array(axisX),*poptExp))
        lomaxR2 = computeR2(percentagemInterArrivalTime,funcLomax(np.array(axisX),*poptLomax))
        lomaxIR2 = computeR2(percentagemInterArrivalTime,funcLomaxI(np.array(axisX),*poptLomaxI))
        weibullR2 = computeR2(percentagemInterArrivalTime,funcWeibull(np.array(axisX),*poptWeibull))

        #coeficientes R2 no dict
        coefR2Lomax[cop]=lomaxR2
        coefR2LomaxI[cop]=lomaxIR2
        coefR2Expo[cop]=expoR2
        coefR2Weibull[cop]=weibullR2
        

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
        distRealPDF[cop]=qtdeInterArrivalTime

        tmpQtde = np.sum(qtdeInterArrivalTime)
        qtdeInterArrivalTimePDF = qtdeInterArrivalTime
        qtdeInterArrivalTime = [float(q)/tmpQtde for q in qtdeInterArrivalTime]
        distRealPDF[cop]=qtdeInterArrivalTime
        
        # comparando o ajuste da PDF e da CDF(funcExponential(t+1),*poptExp)
        
        arrayPDFExponential = []
        arrayPDFLomax = []
        arrayPDFWeibull = []
        for t in np.arange(0,61,1):
            arrayPDFExponential.append((funcExponential((t+1),*poptExp)) - (funcExponential(t,*poptExp)))
            arrayPDFLomax.append((funcLomax((t+1),*poptLomax)) - (funcLomax(t,*poptLomax)))
            arrayPDFWeibull.append((funcWeibull((t+1),*poptWeibull)) - (funcWeibull(t,*poptWeibull)))

        plt.close('all')
        fig = plt.figure()
        fig.suptitle(cop+"\nInter-arrival time")
        plt.plot(
            axisX,qtdeInterArrivalTime,'ro-',#label='Real' ,
            axisX,arrayPDFExponential,'bx-',
            axisX,arrayPDFLomax,'cx-',        
            axisX,arrayPDFWeibull,'gx-',
        )
        plt.ylabel("Quantity [Units]")
        plt.xlabel("Interval [minutes]")
        plt.xticks(axisX,rotation=45)
        plt.grid(True)
        fig.set_size_inches(18.5,10.5)
        plt.legend(prop={'size':16},bbox_to_anchor=(0.99, 0.5))
        #fig.savefig(cop+'/'+'quantity_'+filename+cop+'.png',dpi=96)
        fig.savefig('PDF_Real/'+cop+'.png',dpi=96)
        plt.close('all')

        #print cop
        #print 'Coeficientes Lomax = ',coefDistribuicaoLomax[cop]
        #print poptLomaxPDF
        #print 'Estatisticas ',compute_statistics(interArrivalTime)
        #print 'Media analitica = ',coefDistribuicaoLomax[cop][1]/(coefDistribuicaoLomax[cop][0]-1.0)
        #print '---'*50

        
def computeR2(y, fy):

    #y = percentagemQtde 
    #fy = funcGenPareto(axisX, *poptPareto))
    
    """
    ss_res = np.dot((y - fy),(y - fy))
    ymean = np.mean(y)

    print cop
        print 'Coeficientes Lomax = ',coefDistribuicaoLomax[cop]
        print 'Estatisticas ',compute_statistics(distRealInterArrival[cop])
        print 'Media analitica = ',coefDistribuicaoLomax[cop][1]/(coefDistribuicaoLomax[cop][0]-1.0)
        print '---'*50
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
    sizeData, (minimum,maximum),arithmeticMean,variance,skeness,kurtosis = stats.describe(arrayErro)
    return (minimum,maximum,arithmeticMean,variance)

def erroMedioAbsoluto(real,ajuste):

    # calcula a media da diferenca, em modulo, entre y e ajuste
    erro = 0
    arrayErro=[]
    for reali, ajustei in zip(real,ajuste):
        arrayErro.append(abs(reali-ajustei))
        
    sizeData, (minimum,maximum),arithmeticMean,variance,skeness,kurtosis = stats.describe(arrayErro)
    return (maximum,arithmeticMean,arrayErro.index(np.amax(arrayErro)))

def erroMedioRelativo(real,ajuste):

    # calcula a media da diferenca, em modulo, entre y e ajuste
    erro = 0
    arrayErro=[]
    for reali, ajustei in zip(real,ajuste):
        if(reali == 0):
            arrayErro.append(abs(reali-ajustei))
        else:
            arrayErro.append(abs(reali-ajustei)/reali)
        
    sizeData, (minimum,maximum),arithmeticMean,variance,skeness,kurtosis = stats.describe(arrayErro)
    return (maximum,arithmeticMean,arrayErro.index(np.amax(arrayErro)))

def invFuncWeibull(y,a,b):

    return a*np.power((-np.log(1-y)),1.0/b)


def intervaloConfianca(cop,limiteTempo,coeficientes,confidence = 0.99):
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

        # controlando por tempo
        
        while tempoSimulacao < limiteTempo:
            #tempoSimulacao = tempoSimulacao + invFuncWeibull(np.random.uniform(0,1),*coeficientes)
            #serieSimulacao[simulacao].append(tempoSimulacao)
            
            randomNumber = random.uniform(0,1)
            ts = 0
            while ts < 60:
                if(funcWeibull(ts,*coeficientes)>randomNumber):
                    tempoSimulacao = tempoSimulacao + ts
                    serieSimulacao[simulacao].append(tempoSimulacao)
                    ts = 61
                ts = ts + 1
        """
        # por qtde
        for item in range(0,limiteTempo+1):
            serieSimulacao[simulacao].append(invFuncWeibull(np.random.uniform(0,1),*coeficientes))
        """
        # calculo do intervalo entre chegadas
        for i in range(0,len(serieSimulacao[simulacao])-1):
            serieIntervaloEntreChegadas[simulacao].append(serieSimulacao[simulacao][i+1] - serieSimulacao[simulacao][i])

        print 'QTDE======= ',len(serieIntervaloEntreChegadas[simulacao])
        
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
        
        #dados da populacao
        sizeData, (minimum,maximum),arithmeticMean,variance,skeness,kurtosis = stats.describe(valoresPorInstante[tempo])
        #l,u = norm.interval(confidence, loc=arithmeticMean, scale=sqrt(variance))

        #media.append(m)
        tmp = 2.58 * sqrt(variance)/sqrt(len(valoresPorInstante[tempo]))
        l = arithmeticMean - tmp
        u = arithmeticMean + tmp
        media.append(arithmeticMean)
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
        Loop principal
    """
    
    allCops = get_available_cops()

    #allCops = set([changeCop(c) for c in get_available_cops()])  

    allActionsDict = get_dict_all_actions()
    allIncidentsDict = get_dict_all_incidents()
    allReportsDict = get_dict_all_reports()
    allIncidentsReportsDict = {}

    #cops para os quais sao criados os graficos  
    graphicsFromCops = [
                        #'CCDA - BSB',
                        'CCDA - BHZ',
                        'CCDA - FOR',
                        'CCDA - REC',
                        'CCDA - RIO',
                        'CCDA - SSA',
                        'CCDA - SAO',
                        'CCDA - POA', 
                        'CCDA - CGB',
                        'CCDA - CTB',
                        'CCDA - MAO',
                        'CCDA - NAT'
                    ]
        
    for cop in graphicsFromCops:
        greatestDate[cop] = np.amax([get_actions_greatest_date(allActionsDict[cop]),
                             get_incidents_greatest_date(allIncidentsDict[cop]),
                             get_reports_greatest_date(allReportsDict[cop])])
        axisXCop[cop] = []
    greatestDate['TODOS'] = terminoAmostragem
    axisXCop['TODOS'] = []
    incidentsSerie = {}
    actionsSerie = {}
    reportsSerie = {}
    incidentsReportsSerie = {}

    incidentsSerie['TODOS'] = []
    actionsSerie['TODOS'] = []
  
    reportsSerie['TODOS'] = []
    incidentsReportsSerie['TODOS'] = []
    for day in matchDays:
        if(day <= greatestDate['TODOS']): #apenas os dias dentro do limite
            incidentsSerie['TODOS'].append(len(get_incidents_near_date(allIncidentsDict['TODOS'],day)))
            actionsSerie['TODOS'].append(len(get_actions_near_date(allActionsDict['TODOS'],day)))
            reportsSerie['TODOS'].append(len(get_reports_near_date(allReportsDict['TODOS'],day)))
            axisXCop['TODOS'].append(day)

    
    # agrupar incidentes e relatos
    incidentsReportsSerie['TODOS'] = [i + r for i,r in zip(incidentsSerie['TODOS'],reportsSerie['TODOS'])]
    allIncidentsReportsDict['TODOS'] = allIncidentsDict['TODOS'] + allReportsDict['TODOS']
    
    #for cop in allCops: 
    for cop in graphicsFromCops: 
        incidentsSerie[cop]=[]
        actionsSerie[cop]=[]
        reportsSerie[cop] = []
        incidentsReportsSerie[cop] = []
        print cop, ' = ',greatestDate[cop]
        for day in matchDays:
            if(day <= greatestDate[cop]): #apenas os dias dentro do limite
                incidentsSerie[cop].append(len(get_incidents_near_date(allIncidentsDict[cop],day)))
                actionsSerie[cop].append(len(get_actions_near_date(allActionsDict[cop],day)))
                reportsSerie[cop].append(len(get_reports_near_date(allReportsDict[cop],day)))
                # agrupar incidentes e relatos
                incidentsReportsSerie[cop] = [i + r for i,r in zip(incidentsSerie[cop],reportsSerie[cop])]
                allIncidentsReportsDict[cop] = allIncidentsDict[cop] + allReportsDict[cop]
                axisXCop[cop].append(day)
    # termino da geracao dos dados para estatisticas
    
    #cop = 'CCDA - BSB'
    #plot_resume_cop("Resumo_"+cop+".png",cop,axisXCop[cop],actionsSerie[cop],incidentsSerie[cop],reportsSerie[cop])

    # inicio da criacao dos graficos

    interArrrival_time_distribution('TODOS',allIncidentsReportsDict['TODOS'], nbins=60,limit =  1 * 3600) # unidade em segundos
    #info_distribution('Distribuicao de Info por horas','TODOS',allIncidentsReportsDict['TODOS'], nbins=24,limit = 24*3600,cor='green')
    plot_resume_cop("Resumo_TODOS.png",'TODOS',axisXCop['TODOS'],actionsSerie['TODOS'],incidentsSerie['TODOS'],reportsSerie['TODOS'])
    
    for cop in graphicsFromCops:        
        interArrrival_time_distribution(cop,allIncidentsReportsDict[cop], nbins=60,limit = 1 * 3600) # unidade em segundos
        #info_distribution('Distribuicao de Info por horas',cop,allIncidentsReportsDict[cop], nbins=24,limit = 24*3600,cor='green')
        plot_resume_cop("Resumo_"+cop+".png",cop,axisXCop[cop],actionsSerie[cop],incidentsSerie[cop],reportsSerie[cop])

    # Dados finais  
    #apresentado R2
    print '*'*120
    print 'TODOS', ' Expo = ', coefR2Expo['TODOS'], ' Lomax I = ',coefR2LomaxI['TODOS'], ' Lomax II = ', coefR2Lomax['TODOS'], ' Weibull = ', coefR2Weibull['TODOS']
    for cop in graphicsFromCops:
        print cop, ' Expo = ', coefR2Expo[cop], ' Lomax I = ',coefR2LomaxI[cop], ' Lomax II = ', coefR2Lomax[cop], ' Weibull = ', coefR2Weibull[cop]

    print '*'*120
    for cop in graphicsFromCops:
        print cop, ' Correlacao = ',correlacao[cop]
    print 'Correlacao Todos = ',correlacao['TODOS']
    print '*'*120

    #graphicsFromCops.append('TODOS')
    
    #PDF de todas juntas 
    print '-'*100
    tmp = []
    for cop in graphicsFromCops:
        tmp.append(distRealPDF[cop])
    plot_interArrival(tmp,graphicsFromCops,['bo-','y^-','gs-','cp-','m*-','kh-','b+-','yD-','g|-','c1-','m3-'],'allPDF_qtde_prob.png','Intervalo entre chegadas',21)

    #plotando as dist reais x Lomax de cada Cops

    for cop in graphicsFromCops:
        plot_interArrival([distRealInterArrival[cop],funcExponential(range(0,61),coefDistribuicaoExpo[cop])],['Real','Exponential'],['ro-','b*-'],cop+'/Real_Expo.png','Real x Exponential')
        plot_interArrival([distRealInterArrival[cop],funcLomax(range(0,61),coefDistribuicaoLomax[cop][0],coefDistribuicaoLomax[cop][1])],['Real','Lomax'],['ro-','g*-'],cop+'/Real_Lomax.png','Real x Lomax')
        plot_interArrival([distRealInterArrival[cop],funcWeibull(range(0,61),coefDistribuicaoWeibull[cop][0],coefDistribuicaoWeibull[cop][1])],['Real','Weibull'],['ro-','c*-'],cop+'/Real_Weibull.png','Real x Weibull')
    #plot_interArrival([distRealInterArrival['TODOS'],funcLomax(range(0,61),coefDistribuicaoLomax['TODOS'][0],coefDistribuicaoLomax['TODOS'][1])],['Real','Lomax II'],['ro-','gs-'],'TODOS/Real_Lomax.png','Real x Lomax')
    
    #CDF de todas juntas 
    print '-'*100
    tmp = []
    for cop in graphicsFromCops:
        tmp.append(distRealInterArrival[cop])
    plot_interArrival(tmp,graphicsFromCops,['bo-','y^-','gs-','cp-','m*-','kh-','b+-','yD-','g|-','c1-','m3-'],'allCDF.png','CDF',21)

        # Gerando tabela de estatistica
    print 'Quantidade de dados'
    for cop in graphicsFromCops:
        print cop,'|',len(allActionsDict[cop]),'|',len(allIncidentsDict[cop]),'|',len(allReportsDict[cop]),'|',len(allIncidentsDict[cop])+len(allReportsDict[cop])
        qtdeEventos[cop] = len(allIncidentsDict[cop])+len(allReportsDict[cop])
    
    # fazendo o estudo A.f(X) + b = r(x)
    # calculo das constantes alpha e beta de ajuste
    print '-'*120
    print 'Correlacao entre modelo e real'
    for cop in graphicsFromCops:
        print cop
        fLomax = funcLomax(np.arange(0,61,1),coefDistribuicaoLomax[cop][0],coefDistribuicaoLomax[cop][1])
        fExpo = funcExponential(np.arange(0,61,1),coefDistribuicaoExpo[cop][0])
        fWeibull = funcWeibull(np.arange(0,61,1),coefDistribuicaoWeibull[cop][0],coefDistribuicaoWeibull[cop][1])
        
        plot_interArrival([distRealInterArrival[cop],fExpo,fLomax,fWeibull],['Real','Exponential','Lomax','Weibull'],['ro-','bs-','g^-','c*-'],cop+'/Comparacao.png','Best Fit')

        #print stats.pearsonr(distRealInterArrival[cop],fExpo)[0]
        print 'Expo = ',stats.pearsonr(distRealInterArrival[cop],fExpo)[0], ' R2 = ', coefR2Expo[cop]
        print 'Lomax = ',stats.pearsonr(distRealInterArrival[cop],fLomax)[0],' R2 = ', coefR2Lomax[cop]
        print 'Weibull = ',stats.pearsonr(distRealInterArrival[cop],fWeibull)[0],' R2 = ', coefR2Weibull[cop]
        #print 'Coeficiente Weibull = ', coefDistribuicaoWeibull[cop]
        
        #limiteTempo = ((greatestDate[cop] - inicioAmostragem).total_seconds())/60.0
        #series, media, lower, upper = intervaloConfianca(cop,limiteTempo,coefDistribuicaoWeibull[cop])

        print 'Comparacao Real Modelo -----'
        for k in [1,2,3,4,5,10,20,30,40,50,60]:
            print 'Real = ', distRealInterArrival[cop][k], ' Modelo = ', fWeibull[k]
        print 'Coeficientes Weibull = ', coefDistribuicaoWeibull[cop]


        # intervalo de amostragem em minutos (último evento - 1o evento)
        #intervalo de confianca
        #limiteTempo = ((greatestDate[cop] - inicioAmostragem).total_seconds())/60.0
        #series, media, lower, upper = intervaloConfianca(cop,limiteTempo,coefDistribuicaoWeibull[cop])
        #plot_interArrival([distRealInterArrival[cop],media,lower,upper],['Real','Mean','Lower','Upper'],['ro-','c*-','c--','c--'],cop+'/IC100_Weibull.png','IC Weibull')
        
        # por qtde
        """
        series, media, lower, upper = intervaloConfianca(qtdeEventos[cop],coefDistribuicaoWeibull[cop])
        plot_interArrival([distRealInterArrival[cop],lower,upper],['Real','Lower','Upper'],['ro-','c*--','c*--'],cop+'QTDEweibull.png','QTDE - IC da Weibull')
        """
        #
        # figuras para o artigo milcom
        #CDF
        plot_interArrival([distRealInterArrival['CCDA - BHZ'],distRealInterArrival['CCDA - SAO']],['COp 2','COp 7'],['cp-','m*-'],'CDF_ComparacaoSAOBHZ.png','Demonstracao',21)
        #PDF
        plot_interArrival([distRealPDF['CCDA - BHZ'],distRealPDF['CCDA - SAO']],['COp 2','COp 7'],['cp-','m*-'],'PDF_ComparacaoSAOBHZ.png','Demonstracao',21)