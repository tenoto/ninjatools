#!/usr/bin/env python

import os 

import argparse

import yaml

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle

import numpy as np

import scipy
from scipy.ndimage.filters import gaussian_filter

from iminuit import Minuit
from probfit import BinnedChi2, Extended, gaussian, Chi2Regression

__version__ = "1.0"


class Hist1D(object):
	def __init__(self, nbins, xlow, xhigh):
		self.nbins = nbins
		self.xlow  = xlow
		self.xhigh = xhigh
		self.hist, self.edges = np.histogram([], bins=nbins, range=(xlow, xhigh))
		self.bins = (self.edges[:-1] + self.edges[1:]) / 2.

	def fill(self, value):
		hist, edges = np.histogram([value], bins=self.nbins, range=(self.xlow, self.xhigh))
		self.hist += hist

	@property
	def data(self):
		return self.bins, self.hist

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def mm2pix(mm):
	return int(mm * 600.0/25.4)

def fill_value_to_outer(img,fill_value,xmin,xmax,ymin,ymax):
	new_img = np.copy(img)
	for i in range(len(img)):
		for j in range(len(img[i])):
			if i <= xmin or i >= xmax:
				new_img[i][j] = fill_value
			if j <= ymin or j >= ymax:
				new_img[i][j] = fill_value
	return new_img

def get_sum_values(img,xcen,ycen,xwid,ywid,filter_value=999999999):
	# xrange: xcen - xwid / 2 ... xcen + xwid / 2
	summed_value = 0.0
	summed_pixel = 0
	for i in range(len(img)):
		for j in range(len(img[i])):
			if (xcen-0.5*xwid) <= i <= (xcen+0.5*xwid) and (ycen-0.5*ywid) <= j <= (ycen+0.5*ywid):
				if img[i][j] <= filter_value:
					summed_value += img[i][j]
					summed_pixel += 1					
	rect = Rectangle((xcen-0.5*xwid,ycen-0.5*ywid),
		xwid,ywid,fill=False,ec='r',lw=3)
	return (summed_value,summed_pixel,rect)

def analyze_image(yamlfile,outdir='data/210511/output'):
	param = yaml.load(open(yamlfile))

	# 600 dpi
	# 25.4 mm / 600 pix = 0.0423 mm/pix 
	# 80.0 mm <--> 1,891 pix
	# pix2mm = 25.4 / 600.0 

	infile = param['infile']
	basename = os.path.basename(os.path.splitext(infile)[0])

	fout = open('%s/%s.out' % (outdir,basename),'w')
	fout.write("%s\n" % infile)

	# Using Mac preview, extract 1,891 pixel square image from the original scanned file.
	image_raw = np.array(Image.open(infile).convert('L'))

	print(image_raw)
	print(image_raw.shape)

	# prepare a canvas
	fig = plt.figure(figsize=(8,7),tight_layout=True)
	ax = fig.add_subplot(111,title=param['title'])
	ax.set_xlabel('Pixel size (0.0423 mm/pix)')		
	ax.set_ylabel('Pixel size')

	# plot the raw pixel image
	axis_image = plt.imshow(image_raw,aspect='equal',cmap='gist_gray',origin='lower')

	# add 10 mm line 
	xstart=100
	ystart=100
	plt.plot([xstart,xstart+mm2pix(10.0)],[ystart,ystart],"-w")
	plt.text(150,120,"10 mm",color='w')

	# filtered image and get the maximum point 
	image_filtered = gaussian_filter(image_raw,param['gaussian_filter_sigma'])
	image_center = fill_value_to_outer(image_filtered,param['fill_value'],param['fill_edge'],1891-param['fill_edge'],param['fill_edge'],1891-param['fill_edge'])
	beam_center_pos = np.unravel_index(np.argmin(image_center),image_center.shape)
	plt.plot(beam_center_pos[0],beam_center_pos[1],'*r',
		label='Beam center',markersize=20)

	# add grid and color bar
	plt.grid(color='#979A9A', linestyle='--', linewidth=1)
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.08)			
	fig.colorbar(axis_image, cax=cax, label='gray scale (black:0, white:255)')

	# add contour after gaussian smoothing
	axis_contour = ax.contour(gaussian_filter(image_raw,param['gaussian_filter_sigma']),
		cmap="autumn")
	ax.clabel(axis_contour)

	outpdf = '%s/%s_rawimage.pdf' % (outdir,basename)
	fig.savefig(outpdf)

	###### 

	#hist_pix, edges_pix = np.histogram(image_raw.flatten(),
	#	bins=2**7,range=(0,2**7))
	#x_pix = 0.5*(edges_pix[1:]+edges_pix[:-1])
	#fig = plt.figure(figsize=(8,7),tight_layout=True)
	#plt.plot(x_pix,hist_pix,marker='', drawstyle='steps-mid')
	#plt.xlabel('Pixel value')
	#plt.ylabel('Number of pixels')
	#fig.savefig('hist.pdf')

	fig = plt.figure(figsize=(8,7),tight_layout=True)
	ext_gauss = Extended(gaussian)
	bc2 = BinnedChi2(ext_gauss, image_raw.flatten(),
		bins=param['fit_bins'])
	m = Minuit(bc2, mean=param['fit_mean'],
		sigma=param['fit_sigma'], N=param['fit_N'])
	m.migrad() # fit
	m.print_param()
	bc2.draw(m)

	outpdf = '%s/%s_pixfit.pdf' % (outdir,basename)
	fig.savefig(outpdf)

	flat_pixel_mean = m.values[0]
	flat_pixel_sigma = m.values[1]	

	filter_setting = 3 * flat_pixel_sigma
	print("filter_setting: %.1f" % filter_setting)

	image_flatsub = image_raw-flat_pixel_mean

	# prepare a canvas
	fig = plt.figure(figsize=(8,7),tight_layout=True)
	ax = fig.add_subplot(111,title=param['title'])
	ax.set_xlabel('Pixel size (0.0423 mm/pix)')		
	ax.set_ylabel('Pixel size')

	# plot the raw pixel image
	axis_image = plt.imshow(image_flatsub,aspect='equal',cmap='gist_gray',origin='lower')

	# add 10 mm line 
	xstart=100
	ystart=100
	plt.plot([xstart,xstart+mm2pix(10.0)],[ystart,ystart],"-w")
	plt.text(150,120,"10 mm",color='w')

	# filtered image and get the maximum point 
	image_filtered = gaussian_filter(image_flatsub,param['gaussian_filter_sigma'])
	image_center = fill_value_to_outer(image_filtered,param['fill_value'],param['fill_edge'],1891-param['fill_edge'],param['fill_edge'],1891-param['fill_edge'])
	beam_center_pos = np.unravel_index(np.argmin(image_center),image_center.shape)
	plt.plot(beam_center_pos[0],beam_center_pos[1],'*r',
		label='Beam center',markersize=20)

	# add grid and color bar
	plt.grid(color='#979A9A', linestyle='--', linewidth=1)
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.08)			
	fig.colorbar(axis_image, cax=cax, label='gray scale (black:0, white:255)')

	# add contour after gaussian smoothing
	axis_contour = ax.contour(gaussian_filter(image_flatsub,param['gaussian_filter_sigma']),
		levels=[-9*flat_pixel_sigma,-6*flat_pixel_sigma,-3*flat_pixel_sigma,-flat_pixel_sigma,flat_pixel_sigma,3*flat_pixel_sigma],
		cmap="autumn")
	ax.clabel(axis_contour)

	a60_value,a60_pixel,a60rect = get_sum_values(image_flatsub,
		param['center_x'],param['center_y'],mm2pix(param['wide_wid_x']),mm2pix(param['wide_wid_y']),
#		param['center_x'],param['center_y'],mm2pix(45),mm2pix(45),
		filter_value=filter_setting)
	ax.add_patch(a60rect) 
	print("Area60",a60_value,a60_pixel)

	a10_value,a10_pixel,a10rect = get_sum_values(image_flatsub,
		param['center_x'],param['center_y'],mm2pix(param['center_wid_x']),mm2pix(param['center_wid_y']),
		filter_value=filter_setting)
	ax.add_patch(a10rect) 
	print("Area10",a10_value,a10_pixel)

	fout.write("Rect60mm_TotalValue:%d\n" % a60_value)
	fout.write("Rect60mm_TotalPixel:%d\n" % a60_pixel)	
	fout.write("Rect60mm_TotalRate:%.3e\n" % (float(a60_value)/float(a60_pixel)))

	fout.write("Rect10mm_TotalValue:%d\n" % a10_value)
	fout.write("Rect10mm_TotalPixel:%d\n" % a10_pixel)	
	fout.write("Rect10mm_TotalRate:%.3e\n" % (float(a10_value)/float(a10_pixel)))

	fout.write("Rect10mmTo60mm_TotalValueRatio:%.3e\n" % (float(a10_value)/float(a60_value)))
	fout.write("Rect10mmTo60mm_TotalPixelRatio:%.3e\n" % (float(a10_pixel)/float(a10_pixel)))
	fout.write("Rect10mmTo60mm_TotalRateRatio:%.3e\n" % ((float(a10_value)/float(a10_pixel))/(float(a60_value)/float(a60_pixel))))
	fout.close()

	outpdf = '%s/%s_flatsub.pdf' % (outdir,basename)
	fig.savefig(outpdf)

def get_parser():
	"""
	Creates a new argument parser.
	"""
	parser = argparse.ArgumentParser('analyze_beam_pattern.py',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		description="""
		Analyze beam pattern.
		"""
		)
	version = '%(prog)s ' + __version__
	parser.add_argument('--yamlfile', '-i', type=str, 
		help='Input yamlfile')	
	return parser

def main(args=None):
	parser = get_parser()
	args = parser.parse_args(args)
	analyze_image(args.yamlfile)

if __name__=="__main__":
	main()
