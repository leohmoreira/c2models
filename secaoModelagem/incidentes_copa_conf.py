# -*- coding: utf-8 -*- 	
import datetime
import collections
import argparse
import numpy as np
from scipy import special, optimize
import matplotlib.pyplot as plt
import time
from matplotlib.dates import DateFormatter


def read_file(fileName):
    '''
	Esta funcao lê o arquivo e retorna um list com DATA e QTDE
    '''    
    return tuple(open(fileName, 'r'))
    
def dateConverter(data):
	
	return data[:data.index(":")-1]

def qtdIncidents(data):
	
	return data[data.index(":")+2:-1]

if __name__ == "__main__":
    
    inicioCopaConf = datetime.datetime(2013,6,15)
    terminoCopaConf = datetime.datetime(2013,6,30)

    datesFromFile = read_file('clusterizado.c2')
    dates = map(dateConverter, datesFromFile)
    incidentes = map(qtdIncidents, datesFromFile)
    datesIncidents = {}
    for i, v in enumerate(dates):
    	datesIncidents[time.strftime("%Y/%m/%d",time.strptime(v,"%Y/%B/%d"))] = incidentes[i]
    
    sortedDatesIncidents = sorted(datesIncidents)
    axisX = []
    axisY = []
    
    for key in sortedDatesIncidents:
    	tmpDate = datetime.datetime.strptime(key,"%Y/%m/%d")
    	if(inicioCopaConf <= tmpDate and tmpDate<=terminoCopaConf):
    		  		
    	    axisX.append(datetime.datetime.strptime(key,"%Y/%m/%d"))
            axisY.append(datesIncidents[key])

    	    
    fig, ax = plt.subplots()
    ax.set_title("Incidentes na Copa das Confederacoes - Jun 13")
    ax.set_ylabel("Quantidade")
    ax.set_xlabel("Dias")
    ax.plot_date(axisX,axisY,fmt='-ro',xdate=True,ydate=False)
    plt.xticks(axisX,rotation=45)
    ax.xaxis.set_major_formatter(DateFormatter("%d/%B"))
    plt.grid(True)
    #plt.show()
    plt.savefig("plot.png",dpi=96)
    
    
    
    
    
