# -*- coding: utf-8 -*- 

from datetime import datetime
import collections
import argparse
import numpy as np
from scipy import special, optimize
import matplotlib.pyplot as plt
import time
from matplotlib.dates import DateFormatter
from scipy import stats
from pylab import text
import os, sys
lib_path_Pacificador = os.path.abspath('/home/moreira/Projetos/COP/pacificador_cop')
sys.path.append(lib_path_Pacificador)
from incidentes.models import *

#Constantes
dateDistanceLimit = 43200 #(12 horas em segundos)
actionSize = 43200 #(12 horas em segundos)
punctualActionSize = 0 #(1 hora em segundos)
inicioAmostragem = datetime(2013,6,10,0,0,0)
terminoAmostragem = datetime(2013,6,30,23,59,59)

def get_all_cops():
	"""
		Retorna todos os COPs baseado nas sincronizacoes
	"""
	
	allSincronizations = Sincronizacao.get_all()

	return set([sinc.cop_responsavel['id'] for sinc in allSincronizations 
			if ((sinc.inicio >= inicioAmostragem) and (sinc.fim <=terminoAmostragem))
			])

	
def get_dict_all_actions():
	"""
		Retorna todas as ações agrupadas em um dicionário cuja chave é o nome do COP
	"""
	allSincronizations = Sincronizacao.get_all()
	dictionaryAllActions = {}
	allCops = get_all_cops()
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
	allCops = get_all_cops()
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
	allCops = get_all_cops()
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
	allCops = get_all_cops()
	return [i for i in allIncidents 
						if(
							(i['operations_center'] in allCops) and
							(inicioAmostragem <= i.reporting_date and i.reporting_date <=terminoAmostragem)
						)
	]
	
def get_dict_all_incidents():
	"""
		Retorna todos os incidentes agrupados em um dicionário cuja chave é o nome do COP
	"""
	dictionaryAllIncidents = {}
	allCops = get_all_cops()
	
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
	return [report for report in allReports 
			if (inicioAmostragem <= report.data_hora and report.data_hora <=terminoAmostragem)
	]
	
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

    #plt.legend(iter(lineObjects),('Incidentes', 'Acoes'),loc='lower center')
    #plt.legend(iter(lineObjects),('Incidentes', 'Acoes'),bbox_to_anchor=(0., 1.02, 1., .102),loc='lower center',mode="expand", borderaxespad=0)
    plt.legend(iter(lineObjects),(rotulos), borderaxespad=0, bbox_to_anchor=(1.11, 0.5),prop={'size':12})
    #plt.savefig(filename,dpi=96)
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

    index = np.arange(len(axisX))
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

def compute_statistics(serie):
	"""
		Computa as estatísticas de SERIE utilizando stats.describe
	"""
	sizeData, (minimum,maximum),arithmeticMean,variance,skeness,kurtosis = stats.describe(serie)

	print "Size Data  = ",sizeData , "Minimo,Maximo = ",(minimum,maximum), "Média = ", arithmeticMean , "Variância = ", variance

if __name__ == "__main__":
	"""
		Loop principal
	"""
	
	matchDays = [datetime(2013,6,15,0,0,0),datetime(2013,6,16,0,0,0),datetime(2013,6,17,0,0,0),datetime(2013,6,19,0,0,0),datetime(2013,6,20,0,0,0),
                 datetime(2013,6,22,0,0,0),datetime(2013,6,23,0,0,0),datetime(2013,6,26,0,0,0),datetime(2013,6,27,0,0,0),datetime(2013,6,30,0,0,0)]
	
	mdays = [#datetime(2013,6,10),datetime(2013,6,11),datetime(2013,6,12),datetime(2013,6,13),datetime(2013,6,14),
			datetime(2013,6,15),datetime(2013,6,16),datetime(2013,6,17),datetime(2013,6,18),datetime(2013,6,19),
			datetime(2013,6,20),datetime(2013,6,21),datetime(2013,6,22),datetime(2013,6,23),datetime(2013,6,24),
			datetime(2013,6,25),datetime(2013,6,26),datetime(2013,6,27),datetime(2013,6,28),datetime(2013,6,29),datetime(2013,6,30)]

	
	# inicio da geracao dos dados para estatisticas
	allActionsDict = get_dict_all_actions()
	allPunctualActionsDict = get_dict_all_actions_by_type('PONTUAL')
	allIntervalActionsDict = get_dict_all_actions_by_type('INTERVALO')
	allIncidentsDict = get_dict_all_incidents()
	allReports = get_all_reports()
	allCops = get_all_cops()
	incidentsSerie = {}
	actionsSerie = {}
	punctualActionsSerie = {}
	intervalActionsSerie = {}
	reportsSerie = []
	incidentsSerie['TODOS'] = []
	actionsSerie['TODOS'] = []
	punctualActionsSerie['TODOS'] = []
	intervalActionsSerie['TODOS'] = []
	for day in matchDays:
		#for day in mdays:
			incidentsSerie['TODOS'].append(len(get_incidents_near_date(allIncidentsDict['TODOS'],day)))
			actionsSerie['TODOS'].append(len(get_actions_near_date(allActionsDict['TODOS'],day)))
			punctualActionsSerie['TODOS'].append(len(get_actions_near_date(allPunctualActionsDict['TODOS'],day)))
			intervalActionsSerie['TODOS'].append(len(get_actions_near_date(allIntervalActionsDict['TODOS'],day)))
			reportsSerie.append(len(get_reports_near_date(allReports,day)))
	for cop in allCops:
		incidentsSerie[cop]=[]
		actionsSerie[cop]=[]
		punctualActionsSerie[cop] = []
		intervalActionsSerie[cop] = []
		for day in matchDays:
		#for day in mdays:
			incidentsSerie[cop].append(len(get_incidents_near_date(allIncidentsDict[cop],day)))
			actionsSerie[cop].append(len(get_actions_near_date(allActionsDict[cop],day)))
			punctualActionsSerie[cop].append(len(get_actions_near_date(allPunctualActionsDict[cop],day)))
			intervalActionsSerie[cop].append(len(get_actions_near_date(allIntervalActionsDict[cop],day)))
	
	
	# termino da geracao dos dados para estatisticas
	
	# inicio da criacao dos graficos

	# Incidentes e Açoes por dia 
	plot_total('incidentes_actions_todos.png',matchDays,
		incidentsSerie['TODOS'],actionsSerie['TODOS'],
		incidentsSerie['CCDA - RIO'],actionsSerie['CCDA - RIO'],
		incidentsSerie['CCDA - BSB'],actionsSerie['CCDA - BSB'],
		incidentsSerie['CCDA - SSA'],actionsSerie['CCDA - SSA'],
		incidentsSerie['CCDA - REC'],actionsSerie['CCDA - REC'],
		incidentsSerie['CCDA - FOR'],actionsSerie['CCDA - FOR'],
		incidentsSerie['CCDA - BHZ'],actionsSerie['CCDA - BHZ'])

	# Incidentes e Açoes Intervalo por dia 
	"""
	plot_total('incidentes_actionsInterval_todos.png',matchDays,
		incidentsSerie['TODOS'],intervalActionsSerie['TODOS'],
		incidentsSerie['CCDA - RIO'],intervalActionsSerie['CCDA - RIO'],
		incidentsSerie['CCDA - BSB'],intervalActionsSerie['CCDA - BSB'],
		incidentsSerie['CCDA - SSA'],intervalActionsSerie['CCDA - SSA'],
		incidentsSerie['CCDA - REC'],intervalActionsSerie['CCDA - REC'],
		incidentsSerie['CCDA - FOR'],intervalActionsSerie['CCDA - FOR'],
		incidentsSerie['CCDA - BHZ'],intervalActionsSerie['CCDA - BHZ'])
	"""
	
	for cop in allCops:
		#incidentes por COP por dia
		plot_graph_bar("incidentes_"+cop+".png",cop + " - Incidentes",matchDays,incidentsSerie[cop],"Incidentes",'r')
		#acoes por COP por dia
		plot_graph_bar("actions_"+cop+".png",cop + " - Acoes",matchDays,actionsSerie[cop],"Acoes",'b')
		#acoes por COP pontuais por dia
		plot_graph_bar("punctualActions_"+cop+".png",cop + " - Acoes Pontuais",matchDays,punctualActionsSerie[cop],"Acoes",'b')
		#acoes intervalo por COP por dia
		plot_graph_bar("intervalActions_"+cop+".png",cop + " - Acoes Intervalo",matchDays,intervalActionsSerie[cop],"Acoes",'b')
		#Relacao incidentes vs Acoes por COP por dia
		plot_graph("incidentes_actions_"+cop+".png",cop + " - Incidentes & Acoes",matchDays,incidentsSerie[cop],actionsSerie[cop],('Incidentes','Acoes'))
	
	# comparacao entre incidentes e relatos
	#plot_graph("increl.png","Incidentes e Relatos",matchDays,incidentsSerie['TODOS'],reportsSerie,('incidentes','relatos'))

	#Relatos de situacao por dia
	plot_graph_bar("incidentes.png","Incidentes",matchDays,incidentsSerie['TODOS'],"Incidentes",'r')

	#Incidentes por dia
	plot_graph_bar("relatosDeSituacao.png","Relatos de Situacao",matchDays,reportsSerie,"Relatos",'g')

	# Dados finais

	print '-' * 100
	print "Total de incidentes", len(allIncidentsDict['TODOS'])
	compute_statistics(incidentsSerie['TODOS'])
	print "Total de ações", len(allActionsDict['TODOS'])
	compute_statistics(actionsSerie['TODOS'])
	print "Total de ações pontuais", len(allPunctualActionsDict['TODOS'])
	compute_statistics(punctualActionsSerie['TODOS'])
	print "Total de ações intervalo", len(allIntervalActionsDict ['TODOS'])
	compute_statistics(intervalActionsSerie['TODOS'])
	print '-' * 100
	

	for cop in allCops:
		print '-' * 100
		print cop
		print "Total de incidentes", len(allIncidentsDict[cop])
		compute_statistics(incidentsSerie[cop])
		print "Total de ações", len(allActionsDict[cop])
		compute_statistics(actionsSerie[cop])
		print "Total de ações pontuais", len(allPunctualActionsDict[cop])
		compute_statistics(punctualActionsSerie[cop])
		print "Total de ações intervalo", len(allIntervalActionsDict [cop])
		compute_statistics(intervalActionsSerie[cop])		