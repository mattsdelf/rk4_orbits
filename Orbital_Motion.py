#!usr/env/bin python3
'''
Orbital_Motion.py
python 3
Matthew Delfavero, Monica Rizzo, Caitlin Rose

This body of code exists to implement rk4 techniques for bodies
in a given potential.
'''
from __future__ import print_function, division, unicode_literals

######## Functions ########

def cart_to_sphr(x,y,z):
	import numpy as np
	r = np.sqrt(x**2 + y**2 + z**2)
	theta = np.arccos(z/r)
	phi = np.arctan2(y,x)
	return r, theta, phi

def sphr_to_cart(r,theta,phi):
	import numpy as np
	x = r*np.sin(theta)*np.cos(phi)
	y = r*np.sin(theta)*np.sin(phi)
	z = r*np.cos(theta)
	return x,y,z

def BT_fig1(obj):
	'''
	Phi(r) = (-GM)/(b + sqrt(b^2 + r^2))
	dPhi(r) = GM r / (sqrt(b^2 + r^2 )*(b + sqrt(b^2 + r^2))^2)
	'''
	import numpy as np
	G = 1.0
	M = 1.0
	b = obj[0]
	x = obj[1]
	y = obj[2]
	z = obj[3]
	vx = obj[4]
	vy = obj[5]
	vz = obj[6]

	#r, theta, phi = cart_to_sphr(x,y,z)
	#vr, vtheta, vphi = cart_to_sphr(vx,vy,vz)

	#ar = -G*M*r/(np.sqrt(b**2 + r**2)*(b + np.sqrt(b**2 + r**2))**2)
	ax = -G*M*x/(np.sqrt(b**2 + x**2)*(b + np.sqrt(b**2 + x**2))**2)
	ay = -G*M*y/(np.sqrt(b**2 + y**2)*(b + np.sqrt(b**2 + y**2))**2)
	az = -G*M*z/(np.sqrt(b**2 + z**2)*(b + np.sqrt(b**2 + z**2))**2)
	#atheta = 0.0
	#aphi =0.0
	#ax, ay, az = sphr_to_cart(ar,atheta,aphi)

	d_obj = np.empty(7)
	d_obj[0] = 0
	d_obj[1] = obj[4]
	d_obj[2] = obj[5]
	d_obj[3] = obj[6]
	d_obj[4] = ax
	d_obj[5] = ay
	d_obj[6] = az

	return d_obj


def grav(obj):
	'''
	This is a gravitational potential.
	Also, I was messing around with my animation tool, using this
	piece of code.
	
	Object description:
		Obj[0] = constant, was using mass. Doesn't do anything right now.
		Obj[1-3] Position
		Obj[4-6] Velocities

	Derivative description:
		d_obj[0] = 0 #constant does not change
		d_obj[1-3] = obj_[4-6] # Update position with velocity
		d_obj[4-6]: actually calculate the accelleration on the body.
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
			
def rk4_step(obj,dt,potential):
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
			integrator = rk4_step
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

def adaptive_loop_writer(
		obj, 
		t_initial, 
		t_final,
		dt_initial, 
		dt_min, 
		desired_fractional_error,
		data_name,
		potential = grav, 
		integrator = rk4_step
		):
	'''
	This function loops through all of the steps, integrating
	with whichever function is specified.
	This function saves the data line by line to a file.
	Inputs: object, initial time, desired final time, 
		initial time step, minimum time step, the desired fractional
		error, the name of the file for the data to be stored in,
		the derivative function, and the integrator function.
	'''
	import numpy as np

	if t_final < t_initial:
		raise(
			Exception("Please choose different final/initial steps"))

	# intialize variables
	t = 0  # time
	e0 = desired_fractional_error # Shorter notation
	dt = dt_initial
	
	# Open the file
	fout = open(data_name,'w')

	# loop through for each step
	while t < t_final:
		# Double time step
		obj_a = np.copy(obj)
		obj_a = integrator(obj_a,dt*2.0,potential)

		# Two single timesteps
		obj_b = np.copy(obj)
		obj_b = integrator(obj_b,dt,potential)
		obj_c = integrator(obj_b,dt,potential)

		# Error-able indecies
		e_ind = np.nonzero(obj_b)
		# calculate fractional error 
		ef = np.sqrt(np.sum(
			((obj_c[e_ind] - obj_a[e_ind])/obj_b[e_ind])**2)
			)

		# Keep time step and increase
		if ef <= e0:

			# handle increment and update values
			t += 2*dt
			obj = obj_c
			write_line(fout, obj, t)

			# If the error is zero, make the timestep bigger.
			# We need to catch ef = 0 because it will cause a divide
			# by zero error if we don't.
			if ef == 0.0:
				dt = 15*dt
			else:
				# Increase time step by 5 or scale factor; 
				# whichever is less
				dt = dt*np.min([5,(0.95*(np.abs(e0/ef)**(0.2)))])

		# discard time step and try again
		else:
			dt_new = 0.95*dt*(np.abs(e0/ef)**(0.25))

			# Check if the new time step is bigger than our minimum
			if dt_new < dt_min:
				print("minimum step size encountered")
				dt = dt_min
			else:
				dt = dt_new

		# Ensure that we do not pass t_final
		if t+2*dt > t_final and t!=t_final:
			dt = (t_final - t)/2

	# close the file
	fout.close()

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


def plot_xy(data_file,image_file,**kwargs):
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
	x = data[:,2]
	y = data[:,3]
	z = data[:,4]
	vx = data[:,5]
	vy = data[:,6]
	vz = data[:,7]

	# Open a figure
	fig, ax = pl.subplots()

	# Stylistic preference
	fig.set_facecolor('grey')
	
	# Make the plot
	ax.plot(x,y,label = "y")
	# kwargs are used to set things like the title and axis labels
	# from a higher fuction level.
	ax.set(**kwargs)
	# Save the image.
	pl.savefig(image_file)
	# close the figure
	pl.close()

######## Main ########

def main():
	import numpy as np
	obj = np.asarray([0.1,1,0,0,0,2.7,0]).astype(float)
	t_initial = 0.0
	t_final = 100.0
	dt_initial = 10**(-1)
	dt_min = 10**(-4)
	desired_fractional_error = 10**(-3)
	data_name = "sample_data.txt"
	image_file = "fig_1.png"

	adaptive_loop_writer(
		obj, 
		t_initial, 
		t_final,
		dt_initial, 
		dt_min, 
		desired_fractional_error,
		data_name,
		potential = BT_fig1, 
		integrator = rk4_step
		)

	plot_xy(data_name,image_file)

	pass

######## Execution ########

if __name__ == "__main__":
	main()
