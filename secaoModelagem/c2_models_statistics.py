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
import os, sys
lib_path_Pacificador = os.path.abspath('/home/moreira/Projetos/COP/pacificador_cop')
sys.path.append(lib_path_Pacificador)
from incidentes.models import *

#Constantes
dateDistanceLimit = 43200 #(24 horas em segundos)
actionSize = 86400 #(24 horas em segundos)
punctualActionSize 3600 #(1 hora em segundos)
inicioAmostragem = datetime(2013,6,10)
terminoAmostragem = datetime(2013,7,3)

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
		for action in get_all_actions():
			if (
				(sinc.cop_responsavel['id'] in allCops) and 
				(
					((action.tipo == 'PONTUAL') and (action.inicio >= inicioAmostragem)) or
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
					((action.tipo == 'PONTUAL') and (action.inicio >= inicioAmostragem)) or
					((action.tipo == 'INTERVALO') and (action.inicio >= inicioAmostragem and action.fim <= terminoAmostragem))
				)
			):
				allActions.append(action)
		
	return allActions
	

def get_actions_near_date_backUP(listActions,date,distance):
	"""
	Retorna todas as ações de "listActions" nas quais o valor "date" esteja próximo em "distance" (em segundos) do inicio ou fim da ação
	"""
	return [action for action in listActions 
			if (
				(
					(action.tipo == 'PONTUAL') and 
					(((date - action.inicio).total_seconds() <= distance) and ((date - action.inicio).total_seconds() >= 0))
				) 
				or
				(
					(action.tipo == 'INTERVALO') and 
					(
						((date - action.inicio).total_seconds() <= distance and ((date - action.inicio).total_seconds() >= 0)) and
						#((action.fim - date).total_seconds() <= distance and ((action.fim - date).total_seconds() >= 0))
						(((action.fim - date).total_seconds() >= 0))
					)
				)
			)]

def get_actions_near_date(listActions,date,distance):
	"""
	Retorna todas as ações de "listActions" nas quais o valor "date" esteja próximo em "distance" (em segundos) do inicio ou fim da ação
	"""
	inc = []
	for action in listActions:
		inicio =  datetime.strptime(datetime.strftime(action.inicio,'%Y/%m/%d 23:59:59'),'%Y/%m/%d %H:%M:%S')
		print inicio

		if (
			# a data esta em [inicio - 1h, inicio + 1h]
			(
				(action.tipo == 'PONTUAL') and 
				(abs((date - action.inicio).total_seconds()) <= punctualActionSize)
				
			) 
			or
			(
				# a ação tem duracao de no maximo actionSize e a data está em [inicio,fim]
				(action.tipo == 'INTERVALO') and 
				(
					((action.fim - action.inicio).total_seconds() <= actionSize) and
					((date - action.inicio).total_seconds() >=0)
				)
				
			)
		):
			inc.apppend(action)
	return inc


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


def get_incidents_near_date(listIncidents,date,distance,indicentDateFormatter = '%Y/%m/%d %H:%M:%S'):
	"""
		Retorna todos os incidentes de "listIncidents" nos quais o valor "date" esteja próximo em "distance" (em segundos) do reporting_date
		indicentDateFormatter indica se a data dos incidentes serão agupadas por
			'day' = formato AAAA/MM/DD 
			'full' = formato AAAA/MM/DD hh:mm:ss
	"""
	
	return [incident for incident in listIncidents
			if 
			(
				(abs((datetime.strptime(datetime.strftime(incident.reporting_date,indicentDateFormatter),indicentDateFormatter) - date).total_seconds()) <= distance)
			)
			]

def plot_total(axisX,
                incidents1,actions1,
                incidents2,actions2,
                incidents3,actions3,
                incidents4,actions4,
                incidents5,actions5,
                incidents6,actions6
            ):

    plt.close('all')
    fig = plt.figure()

    graphRIO = plt.subplot2grid((3,2),(0,0))
       
    graphRIO.set_title("RIO DE JANEIRO - Correlacao: " + str(stats.pearsonr(incidents1,actions1)[0]))
    graphRIO.set_ylabel("Quantidade")
    graphRIO.set_xlabel("Dias")
    graphRIO.plot(axisX,incidents1, 'bo-',axisX,actions1, 'ro-')
    graphRIO.xaxis_date()
    graphRIO.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphRIO.grid(True)

    graphBSB = plt.subplot2grid((3,2),(0,1))
    graphBSB.set_title("BRASILIA - Correlacao: " + str(stats.pearsonr(incidents2,actions2)[0]))
    graphBSB.set_ylabel("Quantidade")
    graphBSB.set_xlabel("Dias")
    graphBSB.plot(axisX,incidents2, 'bo-',axisX,actions2, 'ro-')
    graphBSB.xaxis_date()
    graphBSB.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphBSB.grid(True)  

    graphSSA = plt.subplot2grid((3,2),(1,0))
    graphSSA.set_title("SALVADOR - Correlacao: " + str(stats.pearsonr(incidents3,actions3)[0]))
    graphSSA.set_ylabel("Quantidade")
    graphSSA.set_xlabel("Dias")
    graphSSA.plot(axisX,incidents3, 'bo-',axisX,actions3, 'ro-')
    graphSSA.xaxis_date()
    graphSSA.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphSSA.grid(True)  

    graphREC = plt.subplot2grid((3,2),(1,1))
    graphREC.set_title("RECIFE - Correlacao: " + str(stats.pearsonr(incidents4,actions4)[0]))
    graphREC.set_ylabel("Quantidade")
    graphREC.set_xlabel("Dias")
    graphREC.plot(axisX,incidents4, 'bo-',axisX,actions4, 'ro-')
    graphREC.xaxis_date()
    graphREC.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphREC.grid(True)  

    graphFOR = plt.subplot2grid((3,2),(2,0))
    graphFOR.set_title("FORTALEZA - Correlacao: " + str(stats.pearsonr(incidents5,actions5)[0]))
    graphFOR.set_ylabel("Quantidade")
    graphFOR.set_xlabel("Dias")
    graphFOR.plot(axisX,incidents5, 'bo-',axisX,actions5, 'ro-')
    graphFOR.xaxis_date()
    graphFOR.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphFOR.grid(True)

    graphBHZ = plt.subplot2grid((3,2),(2,1))
    graphBHZ.set_title("BELO HORIZONTE - Correlacao: " + str(stats.pearsonr(incidents6,actions6)[0]))
    graphBHZ.set_ylabel("Quantidade")
    graphBHZ.set_xlabel("Dias")
    graphBHZ.plot(axisX,incidents6, 'bo-',axisX,actions6, 'ro-')
    graphBHZ.xaxis_date()
    graphBHZ.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graphBHZ.grid(True)    

    plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
    plt.show()
    #plt.savefig("plot.png",dpi=96)

def plot_graph(axisX, incidents,actions):

    fig, graph1 = plt.subplots()
    graph1.set_title("Incidentes na Copa das Confederacoes - Jun 13")
    graph1.set_ylabel("Quantidade")
    graph1.set_xlabel("Dias")
    graph1.plot(axisX,incidents, 'ro-',axisX,actions, 'bo-')
    graph1.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    plt.xticks(axisX,rotation=90)
    graph1.grid(True)
    plt.show()
    #plt.savefig("plot.png",dpi=96)

if __name__ == "__main__":
	"""
		Loop principal
	"""
	
	matchDays = [datetime(2013,6,15),datetime(2013,6,16),datetime(2013,6,17),datetime(2013,6,19),datetime(2013,6,20),
                 datetime(2013,6,22),datetime(2013,6,23),datetime(2013,6,26),datetime(2013,6,27),datetime(2013,6,30)]
	
	allActionsDict = get_dict_all_actions()
	allIncidentsDict = get_dict_all_incidents()
	
	"""	
	for cop in get_all_cops():
		for days in matchDays:
			print cop," -> ", days, " -> incidents = " , len(get_incidents_near_date(allIncidentsDict[cop],days,dateDistanceLimit,'day')),"acões = ",len(get_actions_near_date(allActionsDict[cop],days,dateDistanceLimit))
	"""
	
	incidentsSerie = {}
	actionsSerie = {}
	
	for cop in get_all_cops():
		incidentsSerie[cop]=[]
		actionsSerie[cop]=[]
		for day in matchDays:
			incidentsSerie[cop].append(len(get_incidents_near_date(allIncidentsDict[cop],day,dateDistanceLimit)))
			actionsSerie[cop].append(len(get_actions_near_date(allActionsDict[cop],day,dateDistanceLimit)))
	"""
	for day in matchDays:
		incRio.append(len(get_incidents_near_date(allIncidentsDict['CCDA - RIO'],day,dateDistanceLimit,'%Y/%m/%d')))
		actRio.append(len(get_actions_near_date(allActionsDict['CCDA - RIO'],day,dateDistanceLimit)))
	"""
	"""
	plot_total(matchDays,
		incidentsSerie['CCDA - RIO'],actionsSerie['CCDA - RIO'],
		incidentsSerie['CCDA - BSB'],actionsSerie['CCDA - BSB'],
		incidentsSerie['CCDA - SSA'],actionsSerie['CCDA - SSA'],
		incidentsSerie['CCDA - REC'],actionsSerie['CCDA - REC'],
		incidentsSerie['CCDA - FOR'],actionsSerie['CCDA - FOR'],
		incidentsSerie['CCDA - BHZ'],actionsSerie['CCDA - BHZ'])
	"""


	print "RIO - acoes comenca em 2013/6/30 = ", len([act for act in allActionsDict['CCDA - RIO'] 
			if datetime.strptime(datetime.strftime(act.inicio,'%Y/%m/%d'),'%Y/%m/%d') == datetime(2013,6,30)
		])
	print len(get_actions_near_date(allActionsDict['CCDA - RIO'],datetime(2013,6,30),dateDistanceLimit))
	print get_actions_near_date(allActionsDict['CCDA - RIO'],datetime(2013,6,30),dateDistanceLimit)

	


