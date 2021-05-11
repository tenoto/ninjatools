#!/usr/bin/env python

import os
import pandas as pd 
import numpy as np

import argparse
from argparse import ArgumentParser

from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

__author__ = 'Teruaki Enoto'
__version__ = '0.01'
# v0.01 : 2021-05-11 : test sctipt

def get_monitor_data(inputfile):
	print("inputfile: %s" % inputfile)

	column_keywords = [
		'rawtimestr','unixtime','timestamp',
		'V1_measure','V2_measure','V3_measure','V4_measure',
		'I1_measure','I2_measure','I3_measure','I4_measure',
		'V1_set','V2_set','V3_set','V4_set',
		'I1_set','I2_set','I3_set','I4_set',		
		'OPV1','OPV2','OPV3','OPV4',				
		'OCP1','OCP2','OCP3','OCP4',
		'state','status','error']
	rawdata = {}	
	for keyword in column_keywords:
		rawdata[keyword] = []

	with open(inputfile, 'r') as f:
		linenum = 0 
		for line in f:
			if linenum > 2:
				cols = line.split()
				rawdata['rawtimestr'].append(cols[0])
				unixtime = float(cols[0][0:10]) + float(cols[0][10:-1])/1e+8
				rawdata['unixtime'].append(unixtime)
				rawdata['timestamp'].append(datetime.fromtimestamp(unixtime))
				for i in range(4):
					rawdata['V%d_measure' % (i+1)].append(float(cols[i+1]))
					rawdata['I%d_measure' % (i+1)].append(float(cols[i+5]))
					rawdata['V%d_set' % (i+1)].append(float(cols[i+9]))
					rawdata['I%d_set' % (i+1)].append(float(cols[i+13]))
		#df = pd.read_csv(f,
		#	skiprows=[0,1,2],sep=r'\\s+',header=None,
		#	engine='python',usecols=[0,1,2,3,4],
		#	names=column_keywords
		#	#delim_whitespace=True
		#	)
			linenum += 1 

	for keyword in column_keywords:
		rawdata[keyword] = np.array(rawdata[keyword])
	return rawdata 

def plot_monitor_data(inputfile,channel,outbasename=None,
		start=None,stop=None,title=None):

	rawdata = get_monitor_data(inputfile)
	if outbasename != None:
		basename = outbasename
	else:
		basename = os.path.splitext(os.path.basename(inputfile))[0]
	outpdf = '%s.pdf' % basename
	outjpg = '%s.jpg' % basename	
	print("outputpdf: %s" % outpdf)

	timestamp_date = rawdata['timestamp'][0].strftime("%Y/%m/%d")

	channel = int(channel)
	cm = 1/2.54  # centimeters in inches
	fig, axs = plt.subplots(2,1,
		figsize=(16.0*cm,9.0*cm),tight_layout=True,
		sharex=True,gridspec_kw={'hspace':0})
	if title != None:
		axs[0].set_title(title)
	axs[0].plot(rawdata['timestamp'],rawdata['V%d_set' % channel],
		label='V%d_set' % channel, ls='--')		
	axs[0].plot(rawdata['timestamp'],rawdata['V%d_measure' % channel],
		label='V%d_measure' % channel)
	axs[0].set_ylabel('Voltage (V)')	
	axs[0].legend(loc='upper left',borderaxespad=1,fontsize=9,ncol=1)
	axs[1].plot(rawdata['timestamp'],1000.0*rawdata['I%d_set' % channel],
		label='I%d_set' % channel,ls='--')
	axs[1].plot(rawdata['timestamp'],1000.0*rawdata['I%d_measure' % channel],
		label='I%d_measure' % channel)
	axs[1].legend(loc='upper left',borderaxespad=1,fontsize=9,ncol=1)	
	axs[1].set_xlabel('Time (%s)' % timestamp_date)
	axs[1].set_ylabel('Current (mA)')		

	plt.gcf().autofmt_xdate()
	plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
	plt.gca().tick_params(axis='both',direction='in')
	plt.xticks(rotation='vertical')

	if start != None and stop != None:
		axs[0].set_xlim(pd.to_datetime(start),pd.to_datetime(stop))
		axs[1].set_xlim(pd.to_datetime(start),pd.to_datetime(stop))		
	fig.align_ylabels()
	fig.savefig(outpdf)
	fig.savefig(outjpg,dpi=800)	

def get_parser():
	"""
	Creates a new argument parser.
	"""
	parser = argparse.ArgumentParser('plot_kikusui_powersupply',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		description="""
		This script plots the monitored data from the Kikusui power suplly.
		"""
		)
	version = '%(prog)s ' + __version__
	parser.add_argument('--inputfile', '-i', type=str, 
		default="210405_WERC_RadTest_EnotoLabPowerSupply_UnixTime.dat", 
		help='Input data file (unix time assumed)')	
	parser.add_argument('--channel', '-c', type=str, 
		default=1,
		help='Channel to be plotted.')		
	parser.add_argument('--start', type=str, 
		default=None,help='Start timestamp (format: 2021-04-05T14:00:00)')
	parser.add_argument('--stop', type=str, 
		default=None,help='Stop timestamp (format: 2021-04-05T17:00:00)')	
	parser.add_argument('--title', type=str, 
		default=None,help='Title (optional)')
	parser.add_argument('--outbasename', type=str, 
		default=None,help='outbasename for output file (optional)')				
	return parser

def main(args=None):
	parser = get_parser()
	args = parser.parse_args(args)

	plot_monitor_data(args.inputfile,args.channel,
		start=args.start,stop=args.stop,title=args.title,
		outbasename=args.outbasename)

if __name__=="__main__":
	main()



