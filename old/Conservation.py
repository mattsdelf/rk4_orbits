#!usr/env/bin python3
'''
Conservation.py
python 3
Matthew Delfavero
Advanced Computational Methods in Physics

This body of code exists to implement rk4 techniques for bodies
in a given potential, and compare them to Euler and Euler-Cromer
methods.
'''
from __future__ import print_function, division, unicode_literals

######## Functions ########

def grav(obj):
	'''
	This is a gravitational potential.
	I used this for a different project.
	Also, I was messing around with my animation tool, using this
	piece of code.
	'''
	import numpy as np
	# Obj: [0] constant [1-3] x,y,z [4-6] vx, vy, vz
	G = 1
	M = 1
	r2 = obj[1]**2 + obj[2]**2 + obj[3]**2
	A = -(G*M/r2)/np.sqrt(r2)
	return np.asarray([0,obj[4],obj[5],obj[6],
		A*obj[1],A*obj[2],A*obj[3]])

def spring(obj):
	'''
	I'll store the quantity (k/m) in obj[0],
	so in this function, I only need that constant.
	Note that F = ma = -ky -> a = -(k/m)y
	Our object includes:
		obj[0]: the constant
		obj[1]: y position
		obj[2]: y velocity

	The derivatives returned from this function break down as follows:
		d(obj[0]) = 0; The constant does not need to change.
		d(obj[1]) = obj[2]; The change in position is equal to 
			the current velocity.
		d(obj[2]) = -obj[0]*obj[1]; This is where we use the constant
			to calculate and return an accelleration.
	'''
	import numpy as np
	return np.asarray([0,obj[2], -obj[0]*obj[1]])

def pendulum(obj):
	'''
	I'll store the quantity (g/L) in obj[0],
	so in this function, I only need that constant.
	Note that F = ma = -g*sin(theta) -> a = -(g/L)sin(theta)
	Our object includes:
		obj[0]: the constant
		obj[1]: theta position
		obj[2]: theta velocity

	The derivatives returned from this function break down as follows:
		d(obj[0]) = 0; The constant does not need to change.
		d(obj[1]) = obj[2]; The change in position is equal to 
			the current velocity.
		d(obj[2]) = -obj[0]*np.sin(obj[1]); This is where we use the 
			constant to calculate and return an accelleration.
	'''
	import numpy as np
	return np.asarray([0,obj[2], -obj[0]*np.sin(obj[1])])
			
def update_from_potential_euler(obj,dt,potential):
	'''
	This function steps an object forward in time,
	using the given potential function and timestep,
	using the euler method.
	'''

	# Create copy of object in initial state
	obj_0 = obj
	# Calculate the set of derivatives for the object at a given
	# state.
	d_obj_0 = potential(obj_0)
	# Step forward, using that set of derivatives.
	obj = obj_0 + dt*d_obj_0
	return obj


def update_from_potential_euler_cromer(obj,dt,potential):
	'''
	This function steps an object forward in time,
	using the given potential function and timestep,
	using the euler-cromer method.

	This function is a problem child if your object isn't in 
	the form:
		obj[0] = constant
		obj[1] = position
		obj[2] = velocity
	This is because euler-cromer only works as a second order
	ODE solver, and therefore, it would be more difficult to 
	implement a generic form without knowing the form of the data
	you are working with.
	'''
	import numpy as np

	# Create copy of object in initial state
	obj_0 = obj
	# Calculate the set of derivatives for the object at a given
	# state.
	# This is the euler step.
	d_obj_0 = potential(obj_0)

	# The euler-cromer method requires a few extra steps
	# with this setup.
	# first, I will initialize an array for the final set
	# of derivatives.
	d_obj_cromer = np.empty_like(obj_0)
	# The constant is still not changing.
	d_obj_cromer[0] = 0
	# Increment the position using the updated velocity
	d_obj_cromer[1] = obj_0[2] + dt*d_obj_0[2]
	# The velocity updated only by the euler velocity.
	d_obj_cromer[2] = d_obj_0[2]

	# Step forward
	obj = obj_0 + dt*d_obj_cromer
	return obj


def update_from_potential_rk4(obj,dt,potential):
	'''
	This function steps an object forward in time,
	using the given potential function and timestep,
	using the fourth order runge kutta method.
	'''

	# Create copy of object in initial state
	obj_0 = obj
	# Calculate the set of derivatives for the object at a given
	# state.
	# This is the euler step.
	d_obj_0 = potential(obj_0)

	# Calculate the first midpoint, and its derivatives.
	obj_1 = obj_0 + (dt/2)*d_obj_0
	d_obj_1 = potential(obj_1)

	# Calculate the second midpoint, and its derivatives.
	obj_2 = obj_0 + (dt/2)*d_obj_1
	d_obj_2 = potential(obj_2)

	# Calculate the full step value, and its derivatives.
	obj_3 = obj_0 + dt*d_obj_2
	d_obj_3 = potential(obj_3)

	# Step forward with the appropriate weighted average.
	obj = obj_0 + (dt/6)*(d_obj_0 + 2*d_obj_1 + 2*d_obj_2 + d_obj_3)
	return obj

def write_line(fout, obj, t):
	''' Write a line with the time and object attributes'''

	# begin the line with the time
	line = "%3.4f"%(t)

	# Write each attribute of the object
	for i in range(len(obj)):
		line += (" %3.4f"%(obj[i]))

	# End with the newline character
	line += '\n'

	# Write the line.
	fout.write(line)


def loop_writer(obj, T, dt, fname,
		potential = grav, 
		integrator = update_from_potential_rk4
		):
	'''
	This function loops through all of the steps, integrating
	with whichever function is specified.
	This function saves the data line by line to a file.
	'''
	import numpy as np

	# calculate the number of timesteps
	n_steps = int(T/dt)

	# initialize the start time
	t = 0

	# Open the file
	fout = open(fname,'w')

	# loop through for each step
	for i in range(n_steps):
		# write the current status of the object
		write_line(fout, obj, t)
		# take a step
		obj = integrator(obj,dt,potential)
		# increment time
		t += dt
	
	# close the file
	fout.close()

def loop_find_period(obj, dt,
		potential = grav, 
		integrator = update_from_potential_rk4
		):
	'''
	This function loops through all of the steps, but rather
	than saving the data to a file, it returns the period of
	the oscillations.
	'''
	import numpy as np

	# Initialize loop variable
	i = 0

	# Initialize the start time
	t = 0

	# Initialize deck
	# A deck is a data structure like a queue or a stack.
	# It's not a queue because you can access information that's
	# not at the front of the queue.
	# Here I use the data structure, deck, to compare the sign
	# of the velocity to the sign of the velocity in the previous
	# step, to find local maxima
	deck_width = 2
	deck = np.zeros((deck_width,len(obj)))

	# Initialize list of local maxima
	local_maxes = []

	# loop through for each step until we have two maxima
	while np.size(local_maxes) < 2:
		# shift the deck:
		for j in range(deck_width -1,-1,-1):
			deck[j] = deck[j-1]
		# take a step
		obj = integrator(obj,dt,potential)
		# Set the bottom of the deck
		deck[0] = obj
		
		# Compare the sign of the velocity to the 
		# previous step to find the maxima
		if (deck[0,2] < 0) and (deck[1,2] > 0):
			local_maxes = np.append(local_maxes,t)

		# increment time
		t += dt

		# increment loop variable
		i += 1

	# The period of the oscillation is the time between two maxima
	period = local_maxes[1] - local_maxes[0]
	
	return period


def part_2():
	import numpy as np

	# 1 second period
	period = 1. 

	# Omega squared for the system (k/m)
	w2 = ((2*np.pi)/period)**2 

	# we want ten oscillations
	T = 10*period 

	# Start at one meter
	y_initial = 1.0

	# Start with zero velocity
	vy_initial = 0.0
	
	# Initialize object
	obj = np.asarray([w2,y_initial,vy_initial]).astype(float) 

	# Create array of timesteps to test
	dt = 10**(np.arange(0,-4,-1).astype(float))

	# Loop for the euler method
	for i in range(len(dt)):

		print("Processing spring potential with euler method, dt=%e"%(dt[i]))
		# Name the file for this data, using the timestep
		# power plus 10
		name_seed = "spring_euler_" + str(
			int(10 + np.log10(dt[i]))).zfill(3)
		data_name = name_seed + ".dat"
		position_plot_name = name_seed + "_y.png"
		energy_plot_name = name_seed + "_E.png"

		# Call looping function for whole period
		loop_writer(obj,T,dt[i],data_name,
			potential = spring,
			integrator = update_from_potential_euler)
	
		# make plots!
		wdt = np.sqrt(w2)*dt[i]
		plot_y(data_name,position_plot_name, 
			xlabel = "Time (s)",
			ylabel = "Position (m)",
			title = "Euler spring position for w*dt = %e"%(wdt),
			)
		plot_E(data_name,energy_plot_name, 
			xlabel = "Time (s)",
			ylabel = "Energy (J)",
			title = "Euler spring energy for w*dt = %e"%(wdt),
			)

	# Loop for the euler-cromer method
	for i in range(len(dt)):

		print("Processing spring potential with euler_cromer method, dt=%e"%(dt[i]))
		# Name the file for this data, using the timestep
		# power plus 10
		name_seed = "spring_euler_cromer_" + str(
			int(10 + np.log10(dt[i]))).zfill(3)
		data_name = name_seed + ".dat"
		position_plot_name = name_seed + "_y.png"
		energy_plot_name = name_seed + "_E.png"

		# Call looping function for whole period
		loop_writer(obj,T,dt[i],data_name,
			potential = spring,
			integrator = update_from_potential_euler_cromer)
		
		# make plots!
		wdt = np.sqrt(w2)*dt[i]
		plot_y(data_name,position_plot_name, 
			xlabel = "Time (s)",
			ylabel = "Position (m)",
			title = "Euler-Cromer spring position for dt w*dt = %e"%(wdt),
			)
		plot_E(data_name,energy_plot_name, 
			xlabel = "Time (s)",
			ylabel = "Energy (J)",
			title = "Euler-Cromer spring energy for dt w*dt = %e"%(wdt),
			)

	# Loop for the rk4 method
	for i in range(len(dt)):

		print("Processing spring potential with rk4, dt=%e"%(dt[i]))
		# Name the file for this data, using the timestep
		# power plus 10
		name_seed = "spring_rk4_" + str(
			int(10 + np.log10(dt[i]))).zfill(3)
		data_name = name_seed + ".dat"
		position_plot_name = name_seed + "_y.png"
		energy_plot_name = name_seed + "_E.png"

		# Call looping function for whole period
		loop_writer(obj,T,dt[i],data_name,
			potential = spring,
			integrator = update_from_potential_rk4)
	
		# make plots!
		wdt = np.sqrt(w2)*dt[i]
		plot_y(data_name,position_plot_name, 
			xlabel = "Time (s)",
			ylabel = "Position (m)",
			title = "rk4 spring position for w*dt = %e"%(wdt)
			)
		plot_E(data_name,energy_plot_name, 
			xlabel = "Time (s)",
			ylabel = "Energy (J)",
			title = "rk4 spring energy for w*dt = %e"%(wdt)
			)

def part_3():
	import numpy as np
	import scipy.special

	# 1 second initial period
	T0 = 1. 

	# Omega squared for the system (g/L)
	# We can take g = 9.80 m/s^2, and L = g/w2
	w2 = ((2*np.pi)/T0)**2

	# Pick a reasonable timestep for rk4
	dt = 10**(-2)

	# We want a range of thetas between 2 and 178 at every 2 degrees.
	# But we also need to convert to radians
	thetas_deg = np.arange(2,180,2)
	thetas_rad = (np.pi/180)*(thetas_deg.astype(float))

	# Start with zero velocity
	vtheta_initial = 0.0
	
	# initialize period array
	periods = np.zeros_like(thetas_rad)

	print("Generating pendulum periods. This may take some time.")

	# Loop through for every theta, and find the period..
	for i in range(len(thetas_deg)):
		# Initialize object
		obj = np.asarray(
			[w2,thetas_rad[i],vtheta_initial]
			).astype(float) 
		# call the loop to find the period.
		# Using a runge kutta
		periods[i] = loop_find_period(obj,dt,
			potential = pendulum,
			integrator = update_from_potential_rk4)

	# Calculate true period values:
	# (T = (2*T0/pi)*K(m), where m = sin^2(theta_0/2)
	#	source: https://en.wikipedia.org/wiki/Pendulum_(mathematics)
	# K(m) = scipy.ellipk(m)
	m_ellipk = np.sin(thetas_rad/2)**2
	K_ellipk = scipy.special.ellipk(m_ellipk)
	periods_true = (2*T0/np.pi)*K_ellipk


	# Make a plot
	plot_periods(thetas_deg,periods,periods_true,T0,
		"pendulum_periods.png",
		title = "Period of simple pendulum vs starting position",
		xlabel = r"$\theta$ (deg)",
		ylabel = "Period (sec)")

	# Create an array to store the data.
	table_array = np.empty((len(thetas_deg),3))
	table_array[:,0] = thetas_deg
	table_array[:,1] = periods/T0
	table_array[:,2] = periods_true/T0
	# Save the table as a csv.
	table_csv(table_array)

######## Plots ########

def plot_y(data_file,image_file,**kwargs):
	'''
	Here, we're going to plot the y values against t for one set of data.
	'''
	import numpy as np
	import pylab as pl


	# load the text file
	data = np.loadtxt(data_file)

	# interpret the text file the way we've written it.
	t = data[:,0]
	w2 = data[0,1]
	y = data[:,2]
	vy = data[:,3]

	# Open a figure
	fig, ax = pl.subplots()

	# Stylistic preference
	fig.set_facecolor('grey')
	
	# Make the plot
	ax.plot(t,y,label = "y")
	# kwargs are used to set things like the title and axis labels
	# from a higher fuction level.
	ax.set(**kwargs)
	# Save the image.
	pl.savefig(image_file)
	# close the figure
	pl.close()


def plot_E(data_file,image_file,**kwargs):
	'''
	Here, we're going to plot the y values against t for one set of data.
	'''
	import numpy as np
	import pylab as pl


	# load the text file
	data = np.loadtxt(data_file)

	# interpret the text file the way we've written it.
	t = data[:,0]
	w2 = data[0,1]
	y = data[:,2]
	vy = data[:,3]

	# k/m = w2 -> (E/m) = (1/2)w2*y^2 + (1/2)v^2
	# I set k/m earlier without care as to what k or m was
	# Therefore with no loss of generality, m = 1, and k = w2
	E = 0.5*w2*(y**2) + 0.5*(vy**2)

	# Open a figure
	fig, ax = pl.subplots()

	# Stylistic preference
	fig.set_facecolor('grey')
	
	# Make the plot
	ax.plot(t,E,label = "E")
	# kwargs are used to set things like the title and axis labels
	# from a higher fuction level.
	ax.set(**kwargs)
	# Save the image.
	pl.savefig(image_file)
	# close the figure
	pl.close()

def plot_periods(theta,periods,periods_true,T0, image_file,**kwargs):
	'''
	plot periods vs thetas for part 3
	'''
	import numpy as np
	import pylab as pl

	# Open a figure
	fig, ax = pl.subplots()

	# Make the plot
	ax.plot(theta,periods/T0,label = "rk4")
	ax.plot(theta,periods_true/T0,label = "scipy.ellipk")
	ax.legend()

	# kwargs are used to set things like the title and axis labels
	# from a higher function level
	ax.set(**kwargs)

	# Save the image
	pl.savefig(image_file)
	# close the figure
	pl.close()

def table_csv(table):
	'''
	Appearently this is really hard
	'''
	# Open file for the first half of the table
	fout_a = open("table_a.csv",'w')

	# Print the first line
	fout_a.write("Amplitude, T/T0 (rk4), T/T0 (true)\n")

	# Write each attribute of the object
	for i in range(int(len(table)/2)):
		line = ""
		line += ("%1.4f, "%(table[i,0]))
		line += ("%1.4f, "%(table[i,1]))
		line += ("%1.4f"%(table[i,2]))

		# End with the newline character
		line += '\n'
		# Write the line
		fout_a.write(line)

	# close the file
	fout_a.close()

	# Open file for the first half of the table
	fout_b = open("table_b.csv",'w')

	# Print the first line
	fout_b.write("Amplitude, T/T0 (rk4), T/T0 (true)\n")

	# Write each attribute of the object
	for i in range(int(len(table)/2),len(table)):
		line = ""
		line += ("%1.4f, "%(table[i,0]))
		line += ("%1.4f, "%(table[i,1]))
		line += ("%1.4f"%(table[i,2]))

		# End with the newline character
		line += '\n'
		# Write the line
		fout_b.write(line)
	
	# close the file
	fout_b.close()


######## Main ########

def main():
	part_2()
	part_3()

######## Execution ########

if __name__ == "__main__":
	main()
