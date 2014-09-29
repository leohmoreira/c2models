# -*- coding: utf-8 -*- 	
from datetime import datetime
import collections
import argparse
import numpy as np
from scipy import special, optimize
import matplotlib.pyplot as plt
import scipy.stats as sta
import time


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
    	axisX.append(datetime.strptime(key,"%Y/%m/%d"))
    	axisY.append(datesIncidents[key])
    
    plt.plot_date(axisX,axisY,fmt='-',xdate=True,ydate=False)
    plt.xticks(axisX, rotation=45)
    plt.grid(True)
    plt.show()
    plt.savefig("plot.png",dpi=96)
    
    
    
    
    
