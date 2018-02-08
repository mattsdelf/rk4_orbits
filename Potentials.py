#!usr/env/bin python3
'''
Potentials.py
python 3
Matthew Delfavero, Monica Rizzo, Caitlin Rose

This body of code exists to implement rk4 techniques for bodies
in a given potential.

In particular, this is a library of potential forces.
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
	By the way, this is a Schwarzschild black hole.
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

	r2 = x**2 + y**2 + z**2

	#r, theta, phi = cart_to_sphr(x,y,z)
	#vr, vtheta, vphi = cart_to_sphr(vx,vy,vz)

	#ar = -G*M*r/(np.sqrt(b**2 + r**2)*(b + np.sqrt(b**2 + r**2))**2)
	ax = -G*M*x/(np.sqrt(b**2 + r2)*(b + np.sqrt(b**2 + r2))**2)
	ay = -G*M*y/(np.sqrt(b**2 + r2)*(b + np.sqrt(b**2 + r2))**2)
	az = -G*M*z/(np.sqrt(b**2 + r2)*(b + np.sqrt(b**2 + r2))**2)
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

def BT_fig_3_3(obj):
	'''
	obj[0] = q
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

	r2 = x**2 + y**2 
	#Lz = x*vy - y*vx
	Lz = 0.2
	q = 0.9
	v0 = 1

	#r, theta, phi = cart_to_sphr(x,y,z)
	#vr, vtheta, vphi = cart_to_sphr(vx,vy,vz)

	#ar = -G*M*r/(np.sqrt(b**2 + r**2)*(b + np.sqrt(b**2 + r**2))**2)
	ax = -(x*v0**2)/(r2 + (z/q)**2) + (x*Lz**2)/(r2**2) 
	ay = -(y*v0**2)/(r2 + (z/q)**2) + (y*Lz**2)/(r2**2)
	az = -(z*v0**2)/((q**2)*(r2 + (z/q)**2))
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


def BT_fig_3_7(obj):
	import numpy as np
	G = 1.0
	M = 1.0
	rc = 0.5
	v0 = 1
	q = 0.7

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
	#ax = -(x*v0**2)/((rc**2 + x**2 + (y/q)*2 + z**2)**(3/2))
	#ay = -(y*v0**2)/((q**2*(rc**2 + x**2 + (y/q)**2) + z**2)**(3/2))
	#az = -(z*v0**2)/((rc**2 + x**2 + (y/q)*2 + z**2)**(3/2))

	ax = -(v0**2)*x/((rc**2 + x**2 + (y/q)**2 + z**2))
	ay = -(v0**2)*y/((q**2)*(rc**2 + x**2 + (y/q)**2 + z**2))
	az = -(v0**2)*z/((rc**2 + x**2 + (y/q)**2 + z**2))
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

def BT_fig_3_16(obj):
	'''
	Phi(r) = (-GM)/(b + sqrt(b^2 + r^2))
	dPhi(r) = GM r / (sqrt(b^2 + r^2 )*(b + sqrt(b^2 + r^2))^2)
	By the way, this is a Schwarzschild black hole.
	'''
	import numpy as np
	G = 1.0
	M = 1.0
	q = 0.8
	v0 = 1
	rc = 0.03
	Omegab = 1.0
	b = obj[0]
	x = obj[1]
	y = obj[2]
	z = obj[3]
	vx = obj[4]
	vy = obj[5]
	vz = obj[6]


	ax = (x*v0**2)/(rc**2 + x**2 + (y/q)**2) - x*Omegab**2
	ay = (y*v0**2)/((q**2)*(rc**2 + x**2 + (y/q)**2)) - y*Omegab**2
	az = 0

	d_obj = np.empty(7)
	d_obj[0] = 0
	d_obj[1] = obj[4]
	d_obj[2] = obj[5]
	d_obj[3] = obj[6]
	d_obj[4] = ax
	d_obj[5] = ay
	d_obj[6] = az

	return d_obj




def kepler(obj):
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
