#Monica Rizzo, 2017
import numpy as np
import math

#main function
def adaptive_RK(func, t_initial, t_final, t_step_initial, t_step_min, y_0, desired_fractional_error):
   """
   Inputs: function that gives derivative (or derivatives), initial time, desired final time, initial time step,
   vector of quantities to be evolved in time (y0), and the desired fractional error

   Outputs: Arrays of time and evolved quantity values
   """

   if t_final<t_initial:
      print("Please choose different final/initial steps")
      return
   
   #initialize time 
   t_array=np.array([t_initial])   
   t=t_initial
   
   e0=desired_fractional_error
  
   #initialize y array
   if hasattr(y, "__len__"):
      y_array=y_0
   else:
      y_array=np.array([y_0[0]])
   
   #initialize time
   dt=t_step_initial
  

   while t<=t_final:
      y1=y
      y2=y
      #twice a single step
      k1=func(t,y)
      k2=func(t+dt,y+0.5*k1*2*dt)
      k3=func(t+dt,y+0.5*k2*2*dt)
      k4=func(t+2*dt,y+k3*2*dt)

      y1=y1+(1/6.0)*(k1+2*k2+2*k3+k4)*2*dt
       
        #one step two times
      k1=func(t,y)
      k2=func(t+0.5*dt,y+0.5*k1*dt)
      k3=func(t+0.5*dt,y+0.5*k2*dt)
      k4=func(t+dt,y+k3*dt)

      y2=y2+(1/6.0)*(k1+2*k2+2*k3+k4)*dt

      k1=func(t+dt,y2)
      k2=func((t+dt)+0.5*dt,y2+0.5*k1*dt)
      k3=func((t+dt)+0.5*dt,y2+0.5*k2*dt)
      k4=func((t+dt)+dt,y2+k3*dt)

      y2=y2+(1/6.0)*(k1+2*k2+2*k3+k4)*dt


      #calculate fractional error
      e_f=abs(y1-y2)/(abs(y+func(t,y)*dt))
       
      if e_f.size>1:
         e_f=max(e_f[0],e_f[1])   
 

      #keep step and increase
      if e_f<e0:
           if t+2*dt<=t_final:
              if hasattr(y, "__len__"):
                 t_array=np.append(t_array,t+2*dt)
                 y_array=np.vstack((y_array,y1))
              else:
                 t_array=np.append(t_array,t+2*dt)
                 y_array=np.append(y_array,y1)
           t=t+2*dt
           y=y2-(y1-y2)/15
           #increase time step
           if e_f==0.0:
              dt=15*dt
           elif 0.95*abs(e0/e_f)**(0.2)<5 and e_f!=0.0:
               dt=0.95*dt*abs(e0/e_f)**(0.2)
           else:
               dt=5*dt
           if t+2*dt>t_final and t!=t_final:
              dt=(t_final-t)/2

      #discard step and try again
      if e_f>e0:
           if 0.95*dt*abs(e0/e_f)**(0.25)<t_step_min:
              print("minimum step size exceeded")
              return
           else:
              dt=0.95*dt*abs(e0/e_f)**(0.25)
           if t+2*dt>t_final and t!=t_final:
              dt=(t_final-t)/2

   return t_array,y_array

