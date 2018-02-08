"""
AnimateStars.py

Matthew Delfavero

Version 1

Purpose: To provide the tools to take a series of star images, and animate them, creating an mp4.

"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import matplotlib.animation as animation
import math
import Orbital_Motion

def get_objects(fname):
	objs = np.loadtxt(fname)
	return objs

def make_grid_one(fname,i,n,color = 'gray'):
	objs = get_objects(fname)
	one_frame_one_obj(objs[i],n,color)

def one_frame_one_obj(obj,n,color,ax,scale):
	grid = np.zeros((n,n))
	x = ((scale*obj[1]).astype(int) + (n/2)) % n
	y = ((scale*obj[2]).astype(int) + (n/2)) % n
	grid[x,y] = 1
	ax.imshow(grid,cmap = color,interpolation = "gaussian")

def sim_vid(infile,outfile,n_steps,frames,scale):
	sig = 2
	alpha = 0.01
	n = 256
	steps_per_frame = int(n_steps/frames)

	CMAP = "gray"
	Title = "Flying Rocks"
	Artist = "Matthew Del Favero"
	Comment = "I'm Awesome"

	objs = get_objects(infile)
	fig, ax = pl.subplots()
	one_frame_one_obj(objs[0],n,CMAP,ax,scale)

	#FFMpegWriter = animation.writers['ffmpeg']
	FFMpegFileWriter = animation.FFMpegFileWriter
	metadata = dict(title = Title, artist = Artist, comment = Comment)
	writer = FFMpegFileWriter(
		fps = 15, 
		metadata = metadata,
		bitrate = -1)

	with writer.saving(fig,outfile,frames):
		for k in range(frames):
			one_frame_one_obj(
				objs[k*steps_per_frame],n,CMAP,ax,scale)
			writer.grab_frame()
			print("Recorded %d out of %d"%((k+1),frames))

def main():
	obj = np.asarray([np.pi/10,4,0,0,0,-2,0]).astype(float)
	data_file = "sample_data.txt"
	video_file = "sample_video.mp4"
	image_file = "sample_plot.png"
	dt = 0.001
	frames = 150
	scale = 5.0
	T = 50.0
	n_steps = int(T/dt)
	Orbital_Motion.loop_writer(obj,T,dt,data_file,
		potential = Orbital_Motion.BT_fig1)
	#Orbital_Motion.plot_xy_vs_t(data_file,image_file)
	sim_vid(data_file,video_file,n_steps,frames,scale)

if __name__ == "__main__":
	main()
