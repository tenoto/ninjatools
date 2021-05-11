#!/usr/bin/env python

import os
import pandas as pd 
import numpy as np

import astropy.io.fits as fits

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

def plot_monitor_data(rawdata,channel,
		outbasename=None,start=None,stop=None,title=None,
		voltage_max=8.0,current_max=1800):

	outpdf = '%s.pdf' % outbasename
	outjpg = '%s.jpg' % outbasename	
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
	axs[0].set_ylim(-0.5,voltage_max)
	axs[0].set_ylabel('Voltage (V)')	
	axs[0].legend(loc='upper left',borderaxespad=1,fontsize=6,ncol=1)

	axs[1].plot(rawdata['timestamp'],1000.0*rawdata['I%d_set' % channel],
		label='I%d_set' % channel,ls='--')
	axs[1].plot(rawdata['timestamp'],1000.0*rawdata['I%d_measure' % channel],
		label='I%d_measure' % channel)
	axs[1].legend(loc='upper left',borderaxespad=1,fontsize=6,ncol=1)	
	axs[1].set_ylim(-100,current_max)
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

def plot_monitor_data_with_hk(rawdata,channel,hkfits,
		outbasename=None,start=None,stop=None,title=None,
		voltage_max=8.0,current_max=1800):

	outpdf = '%s.pdf' % outbasename
	outjpg = '%s.jpg' % outbasename	
	timestamp_date = rawdata['timestamp'][0].strftime("%Y/%m/%d")

	hdu = fits.open(hkfits)

	timestamp_list = []
	for unixtime_str in hdu['HK'].data['TIME']:
		timestamp_list.append(datetime.fromtimestamp(unixtime_str))
	timestamp_array = np.array(timestamp_list)

	fram_check_list = []
	for fram_str in hdu['HK'].data['FRAM']:
		if fram_str == 'FC00':
			fram_check_list.append(1.0)
		else:
			fram_check_list.append(0.0)
	fram_check_array = np.array(fram_check_list)

	channel = int(channel)
	cm = 1/2.54  # centimeters in inches
	fig, axs = plt.subplots(5,1,
		figsize=(9.0*cm,12.0*cm),tight_layout=True,
		sharex=True,gridspec_kw={'hspace':0})
	if title != None:
		axs[0].set_title(title)
	axs[0].plot(rawdata['timestamp'],rawdata['V%d_set' % channel],
		label='V%d_set' % channel, ls='--')		
	axs[0].plot(rawdata['timestamp'],rawdata['V%d_measure' % channel],
		label='V%d_measure' % channel)
	axs[0].set_ylim(-0.5,voltage_max)
	axs[0].set_ylabel('Volt. (V)')	
	axs[0].legend(loc='upper left',borderaxespad=1,fontsize=6,ncol=1)

	axs[1].plot(rawdata['timestamp'],1000.0*rawdata['I%d_set' % channel],
		label='I%d_set' % channel,ls='--')
	axs[1].plot(rawdata['timestamp'],1000.0*rawdata['I%d_measure' % channel],
		label='I%d_measure' % channel)
	axs[1].legend(loc='upper left',borderaxespad=1,fontsize=6,ncol=1)	
	axs[1].set_ylim(-100,current_max)	
	axs[1].set_ylabel('Curr. (mA)')		

	axs[2].plot(timestamp_array,hdu['HK'].data['VOLT'],"o--",markersize=2.0)
	axs[2].set_ylim(-2.,14)	
	axs[2].set_ylabel('CPU (V)')	

	axs[3].plot(timestamp_array,hdu['HK'].data['TEMP'],"o--",markersize=2.0)
	axs[3].set_ylim(-50,300)
	axs[3].set_ylabel('Temp (C)')		

	axs[4].plot(timestamp_array,fram_check_array,"o--",markersize=2.0)
	axs[4].set_ylim(-0.5,1.5)
	axs[4].set_ylabel('FRAM')			

	axs[-1].set_xlabel('Time (%s)' % timestamp_date)

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

rawdata = get_monitor_data("data/210405_WERC_RadTest_EnotoLabPowerSupply_UnixTime.dat")

"""
plot_monitor_data(rawdata,2,
	title="WERC radiation test, TAC-DAQ power supply",
	outbasename="out/210405_WERC_RadTest_EnotoLabPowerSupply_tacdaq_beam1",
	start="2021-04-05T13:50:00",stop="2021-04-05T17:00:00")

plot_monitor_data(rawdata,2,
	title="WERC radiation test, TAC-DAQ power supply",
	outbasename="out/210405_WERC_RadTest_EnotoLabPowerSupply_tacdaq_beam1a",
	start="2021-04-05T14:00:00",stop="2021-04-05T15:00:00",
	voltage_max=6.0,current_max=1200)

plot_monitor_data(rawdata,2,
	title="WERC radiation test, TAC-DAQ power supply",
	outbasename="out/210405_WERC_RadTest_EnotoLabPowerSupply_tacdaq_beam1b",
	start="2021-04-05T15:00:00",stop="2021-04-05T16:00:00",
	voltage_max=6.0,current_max=1200)
"""

plot_monitor_data_with_hk(rawdata,2,'data/210405_sum.fits',
	title="all span",
	outbasename="out/210405_WERC_RadTest_EnotoLabPowerSupply_tacdaq_hk_all",
	start="2021-04-05T14:00:00",stop="2021-04-05T17:00:00",
	voltage_max=6.0,current_max=1200)

plot_monitor_data_with_hk(rawdata,2,'data/210405_sum.fits',
	title="Event 1",
	outbasename="out/210405_WERC_RadTest_EnotoLabPowerSupply_tacdaq_hk_evt1",
	start="2021-04-05T14:00:00",stop="2021-04-05T14:20:00",
	voltage_max=6.0,current_max=1200)

plot_monitor_data_with_hk(rawdata,2,'data/210405_sum.fits',
	title="Event 2",
	outbasename="out/210405_WERC_RadTest_EnotoLabPowerSupply_tacdaq_hk_evt2",
	start="2021-04-05T15:10:00",stop="2021-04-05T15:30:00",
	voltage_max=6.0,current_max=1200)

plot_monitor_data_with_hk(rawdata,2,'data/210405_sum.fits',
	title="Event 3,4,5,6",
	outbasename="out/210405_WERC_RadTest_EnotoLabPowerSupply_tacdaq_hk_evt3",
	start="2021-04-05T15:30:00",stop="2021-04-05T16:00:00",
	voltage_max=6.0,current_max=1200)


