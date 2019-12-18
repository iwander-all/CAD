########################################
# 11/9/2019
# from HKUST, MESF MSc
# iwander
########################################

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

np.set_printoptions(suppress = True)
#database = []
m = np.array([],dtype = np.uint8) # the number of bazier curves
pointsNumber = np.array([],dtype = np.uint8) #for the ith curve, the points number of it
closedOrOpen = np.array([],dtype = np.uint8) #for the ith curve, 0 means it is open, 1 means it is closed
point = np.array([],dtype = np.float64) #the control points for all the curves
f = np.array([],dtype=np.float64)
u_i = np.array([],dtype = np.float64)#u_i is the current u for current line for current iteration
bezierLine = np.array([],dtype=np.float64) #list of points of path
pointA = np.array([],dtype=np.float64) 
pointB = np.array([],dtype=np.float64)
D = np.array([],dtype=np.float64) #the history of path length
min_D = 10000000 #make sure that D is smaller after each iteration


pointsNumber_str = ""
closedOrOpen_str = ""
point_str = ""
line_zero = ""
filepath = ""


def inputFromText():
  global filepath
  filepath = os.path.abspath('.') + "/" + raw_input("please input the file name(eg:data.txt):")

  with open (filepath) as data:
    line_zero = data.readline()
    line_zero_list = list(line_zero.split())
    global m
    m = np.array(line_zero_list, dtype=np.uint8)

    for i in range(m[0]): 
      pointsNumber_str = data.readline()
      pointsNumber_list = list(pointsNumber_str.split())
      temp1 = np.array(pointsNumber_list,dtype=np.uint8)
      global pointsNumber
      pointsNumber = np.append(pointsNumber,temp1) # the number of control points of the ith Bezier curve

      closedOrOpen_str = data.readline()
      closedOrOpen_list = list(closedOrOpen_str.split())
      temp2 = np.array(closedOrOpen_list,dtype=np.uint8)
      global closedOrOpen
      closedOrOpen = np.append(closedOrOpen,temp2) #0 means the ith curve is closed, 1 means it is open
      
      global point
      for j in range(temp1[0]):
      #u += 1
        point_str = data.readline()
        point_list = list(point_str.split())
        temp3 = np.array(point_list,dtype=np.float64)
        point = np.append(point,temp3) # control points for the ith Bezier Curve

    global pointA
    pointA_str = data.readline()
    pointA_list = list(pointA_str.split())
    temp4 = np.array(pointA_list,dtype=np.float64)
    pointA = np.append(pointA,temp4) # point A 3*1
    global pointB
    pointB_str = data.readline()
    pointB_list = list(pointB_str.split())
    temp5 = np.array(pointB_list,dtype=np.float64)
    pointB = np.append(pointB,temp5) # point B 3*1 
    
    global bezierLine
    bezierLine = np.append(bezierLine,pointA)
    #bezierLine = bezierLine.reshape((-1,3))
  
  point = np.reshape(point, (-1,3))
  #print(m)
  #print(pointsNumber)
  #print(closedOrOpen)
  #print(point)

def getBezierKurve(closedOrOpen,pointsNumber,point,point_index,u_i,jacobian_flag,drawbezier_flag,iter_flag,jacobiancontrol_flag):
#closedOrOpen: current kurve is closed or open bezier kurve
#pointsNumber: the number of control points of current kurve
#point: the xyz coordinates of control points of total kurve
#point_index: help find which control points belong to current kurve
#u_i: the current u for the current iteration
#jacobian_flag: help decide witch part of Jacobian is non-zero

#because this function conbine too many meanings(not good) so we need flags to control which one is on:
#drawbezier_flag: if true, it will draw bezier kurves
#iter_flag: if true, it will draw the current trace points and lines
#jacobiancontrol_flag: if true, it will calculate the partial derivative

  C = []
  n = pointsNumber - 1 # n is fewer in numbers than the total control points number. According to definition.
  point_show = np.array([],dtype=np.float64)
  for i in range(n+1):
    point_show = np.append(point_show,point[point_index + i])  

  if (closedOrOpen == 0): # if it is closed, means the oth and nth control points are the same.
    n += 1
    point_show = np.append(point_show,point[point_index])
  elif (closedOrOpen == 1):
    pass
  point_show = point_show.reshape((-1,3))

  if ((n+1) % 2 == 0):
    for i in range((n+1) / 2):
      up = 1
      down = 1
      j = n
      while (j > i):
        up *= j
        j = j - 1
      j = n - i
      while (j > 0):
        down *= j
        j = j - 1
      C.append(up / down)
  elif ((n+1) % 2 == 1):
    for i in range(n / 2):
      up = 1
      down = 1
      j = n
      while (j > i):
        up *= j
        j = j - 1
      j = n - i
      while (j > 0):
        down *= j
        j = j - 1
      C.append(up / down)
    up = 1
    down = 1
    j = n
    while (j > n/2):
      up *= j
      j = j - 1
    j = n/2
    while (j > 0):
      down *= j
      j = j - 1
    C.append(up / down)
  if (n%2 == 1):
    for i in range(int((n+1)/2)):
      C.append(C[int(n/2-i)])
  if (n%2 == 0):
    for i in range(int((n+1)/2)):
      C.append(C[int(n/2-i-1)])
  #print("C",C) 

  if (drawbezier_flag==1):
    global f
    #f = np.array([],dtype=np.float64)
    #the following steps is to draw the kurve,not so important
  #  print("point_show\n",point_show)
  #  print("n+1",n+1)
    u = 0
    for j in range(1000): 
      fx = 0
      fy = 0  #do not forget return 0!!!!!!!!!!!!
      fz = 0
      for i in range(n+1):
        fx += C[i] * u**i * (1-u)**(n-i) * point_show[i][0]
        fy += C[i] * u**i * (1-u)**(n-i) * point_show[i][1]
        fz += C[i] * u**i * (1-u)**(n-i) * point_show[i][2]
      list = []
      list.append(fx)
      list.append(fy) 
      list.append(fz)
      array_list = np.array([list],dtype=np.float64) 
      u += 0.001
      f = np.append(f,array_list)

    f = f.reshape((-1,3))
    #ax.plot(f_show[:,0],f_show[:,1],f_show[:,2],'b.',markersize=1,label='open bazier curve')
    #ax.plot(point_show[:,0],point_show[:,1],point_show[:,2],'-',markersize=1, label='control line')
    #ax.plot(point_show[:,0],point_show[:,1],point_show[:,2],'r.',markersize=8, label='control points')

  if (iter_flag==1):
    fx = 0
    fy = 0  #do not forget return 0!!!!!!!!!!!!
    fz = 0
    for i in range(n+1):
      fx += C[i] * u_i**i * (1-u_i)**(n-i) * point_show[i][0]
      fy += C[i] * u_i**i * (1-u_i)**(n-i) * point_show[i][1]
      fz += C[i] * u_i**i * (1-u_i)**(n-i) * point_show[i][2]
    list = []
    list.append(fx)
    list.append(fy) 
    list.append(fz)
    array_list = np.array([list],dtype=np.float64) 
    global trace
    trace = np.array([],dtype=np.float64)
    trace = np.append(trace,array_list)

  if (jacobiancontrol_flag==1):
  #the following step is to get the current line for current iteration
    fx = 0
    fy = 0  #do not forget return 0!!!!!!!!!!!
    fz = 0
    for i in range(n+1):
      fx += C[i] * u_i**i * (1-u_i)**(n-i) * point_show[i][0]
      fy += C[i] * u_i**i * (1-u_i)**(n-i) * point_show[i][1]
      fz += C[i] * u_i**i * (1-u_i)**(n-i) * point_show[i][2]
    #u_i is the current u for current line for current iteration  
    _list = []
    _list.append(fx)
    _list.append(fy) 
    _list.append(fz)
    _array_list = np.array([_list],dtype=np.float64)
    global bezierLine #for current u, find the corresponding points on each line
    bezierLine = np.append(bezierLine,_array_list)
    #bezierLine = bezierLine.reshape((-1,3))
    #print("bezierLine")
    #print(bezierLine)

    # calculate the partial derivative
    global partial_derivative
    #partial_derivative = np.zeros(3*m[0]) 
    partial_derivative = np.zeros(3)
    dfx = 0
    dfy = 0  #do not forget return 0!!!!!!!!!!!
    dfz = 0
    for i in range(n+1):
      if (i==0):
        dfx -= C[i] * (n-i)*(1-u_i)**(n-i-1) * point_show[i][0]
        dfy -= C[i] * (n-i)*(1-u_i)**(n-i-1) * point_show[i][1]
        dfz -= C[i] * (n-i)*(1-u_i)**(n-i-1) * point_show[i][2] 
      elif (n==i):
        dfx += C[i] * i*u_i**(i-1) * point_show[i][0]
        dfy += C[i] * i*u_i**(i-1) * point_show[i][1]
        dfz += C[i] * i*u_i**(i-1) * point_show[i][2]
      else: 
        dfx += C[i] * (i*u_i**(i-1)*(1-u_i)**(n-i)-u_i**i*(n-i)*(1-u_i)**(n-i-1)) * point_show[i][0]
        dfy += C[i] * (i*u_i**(i-1)*(1-u_i)**(n-i)-u_i**i*(n-i)*(1-u_i)**(n-i-1)) * point_show[i][1]
        dfz += C[i] * (i*u_i**(i-1)*(1-u_i)**(n-i)-u_i**i*(n-i)*(1-u_i)**(n-i-1)) * point_show[i][2]
    #partial_derivative[jacobian_flag*3] = dfx
    #partial_derivative[jacobian_flag*3+1] = dfy
    #partial_derivative[jacobian_flag*3+2] = dfz
    partial_derivative[0] = dfx
    partial_derivative[1] = dfy
    partial_derivative[2] = dfz
    #print("partial_derivative")
    #print(partial_derivative)


def steepestAscendmethod(ITERATION_TIME):
  #global u_i
  #u_i = np.random.rand(m[0],1)
  #print("u_i\n",u_i)
  #delta_u = np.zeros((m[0],3))
  #u_i = np.mat(u_i)
  #delta_u = np.mat(delta_u)
  #global r #residual = line1 - line2
  #r = np.zeros((m[0]-1,3)) 
  #print("r\n",r)
  #global Jacobian
  #Jacobian = np.zeros((m[0]-1,3*m[0]))
  #print("Jacobian\n",Jacobian)
  D = 0
  unit_step = 0.0005 #learning rate for each iteration
  kurve_jacobian = np.array([],dtype=np.float64) #partial derivative of line df/du
  jacobian_trace = np.zeros((m[0],1)) #jacobian of trace length = [dD/du1 dD/du2 ...  dD/dun]
  #global ax

  fig = plt.figure()
  ax = fig.gca(projection='3d')

  for i in range(ITERATION_TIME):
    point_index = 0
    #key = raw_input("should we continue? yes/no:")
    if (0):#key =="no"):
      #print("OK, you are right.\n")
      #break
      pass
    elif (1):#key =="yes"):
      #print("let's continue:\n")
      point_index = 0

      #fig = plt.figure()
      #ax = fig.gca(projection='3d')

      for i in range(m[0]):
        getBezierKurve(closedOrOpen[i],pointsNumber[i],point,point_index,0,0,1,0,0) #draw curves only
        point_index += pointsNumber[i]

      point_index = 0
      for j in range(m[0]):
        #print(j)
        #print("*******************************")
        getBezierKurve(closedOrOpen[j],pointsNumber[j],point,point_index,u_i[j],0,0,0,1)
        point_index += pointsNumber[j]
        #temp1 = np.zeros(3*m[0]) 
        #temp1 = partial_derivative
        kurve_jacobian = np.append(kurve_jacobian,partial_derivative) # do not forget reshape

      kurve_jacobian = kurve_jacobian.reshape((-1,3))
      #print("kurve_jacobian")
      #print(kurve_jacobian.shape)
      #print(kurve_jacobian)
      global bezierLine
      bezierLine = np.append(bezierLine,pointB)
      bezierLine = bezierLine.reshape((-1,3))
      #print("bezierLine")
      #print(bezierLine.shape)
      #print(bezierLine)

      ax.plot(bezierLine[:,0],bezierLine[:,1],bezierLine[:,2],'-',markersize=1, label='trace')
      ax.plot(f[:,0],f[:,1],f[:,2],'r.',markersize=1)#,label='open bazier curve')
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
      plt.show()

      for i in range(m[0]):
        jacobian_trace[i] = 2*((bezierLine[i+1][0]-bezierLine[i+2][0])-(bezierLine[i][0]-bezierLine[i+1][0]))*kurve_jacobian[i][0] + 2*((bezierLine[i+1][1]-bezierLine[i+2][1])-(bezierLine[i][1]-bezierLine[i+1][1]))*kurve_jacobian[i][1] + 2*((bezierLine[i+1][2]-bezierLine[i+2][2])-(bezierLine[i][2]-bezierLine[i+1][2]))*kurve_jacobian[i][2]
      #print(jacobian_trace)
      
      global D
      D_item = 0
      for i in range(m[0]):
        D_item += ((bezierLine[i+1][0]-bezierLine[i+2][0])-(bezierLine[i][0]-bezierLine[i+1][0]))**2 + ((bezierLine[i+1][1]-bezierLine[i+2][1])-(bezierLine[i][1]-bezierLine[i+1][1]))**2 + ((bezierLine[i+1][2]-bezierLine[i+2][2])-(bezierLine[i][2]-bezierLine[i+1][2]))**2
      D = np.append(D,D_item)
      print("the length of trace")
      #print(D_item)      

      if (D[D.shape[0]-1] <= min_D):      
        for i in range(m[0]):
          step = jacobian_trace[i] * unit_step
          u_i[i] -= step
          while (u_i[i] > 1 or u_i[i] < 0): #make sure that 0<u<1
            u_i[i] += step
            step = step * unit_step
            u_i[i] -= step   
      #print("step")
      #print(step)
      #print("u_i")
      #print(u_i)

      global min_D
      if (min_D > D[D.shape[0]-1]):
        min_D = D[D.shape[0]-1]
      else:
        D[D.shape[0]-1] = min_D
      print(D[D.shape[0]-1])


      kurve_jacobian = np.array([],dtype=np.float64) #return to init station
      bezierLine = np.array([],dtype=np.float64)
      bezierLine = np.append(bezierLine,pointA)
      #for i in range(m[0]-1):
      #  np.delete(bezierLine,m[0]-1-i,axis = 0)

      #key = raw_input("should we close the frame? yes/no:")
      #if (key == "yes"):
      #  plt.clf()
      #  plt.ioff() 

        #print("temp1\t")
        #print(temp1)
        #getBezierKurve(closedOrOpen[j+1],pointsNumber[j+1],point,point_index,u_i[j+1],j+1,1,0,1)
        #point_index += pointsNumber[j+1]
        #temp2 = np.zeros(3*m[0]) 
        #temp2 = partial_derivative
        #print("temp2\t")
        #print(temp2)
        #temp = np.zeros(3*m[0]) 
        #temp = temp1 - temp2 #this is the jacobian for current residual r[j](shape:1*3m[0])
        #print("temp\t")
        #print(temp)
        #print("bezierLine\n")
        #print(bezierLine)
        #r[j] = bezierLine[2*j] - bezierLine[2*j+1]  #risiduals: because same value is pushed twice,so we need 2
        #print("r\n")
        #print(r)
        #Jacobian[j] = temp
        #print("Jacobian\n")
        #print(Jacobian)
      #J = np.mat(Jacobian)
      #J_transpose = J.T
      #print("Jacobian_T\n")
      #print(J_transpose.shape)
      #H = np.dot(J_transpose,J)
      #print("H\n")
      #print(H.shape)
      #b = J_transpose * r
      #print(b.shape)
      #H_rev = np.linalg.pinv(H)
      #print(H_rev.shape)
      #r = np.mat(r)
      #delta_u = np.linalg.lstsq(np.dot(Jacobian.T,Jacobian), -np.dot(Jacobian.T,r))
      #delta_u = np.linalg.lstsq(H, -np.dot(Jacobian.T,r))
      #delta_u = - H_rev * J_transpose * r
      #print("delta_u")
      #print(delta_u)
      # this is gauss-newton method
      #u_i += delta_u
      #print("updated_u")
      #print(u_i)

      #now,let's draw these points and lines to show the iteration!
      #for i in range(m[0]):
      #  getBezierKurve(closedOrOpen[i],pointsNumber[i],point,point_index,0,i,0,1,0)
      #  point_index += pointsNumber[i]

      #trace = trace.reshape((-1,3))
      #ax.plot(trace[:,0],trace[:,1],trace[:,2],'g.',markersize=8)
      #ax.plot(trace[:,0],trace[:,1],trace[:,2],'-','p.')

    else:
      print("Error: unknown error")



# main =>
inputFromText()
#fig = plt.figure()
#ax = fig.gca(projection='3d')

u_i = np.random.rand(m[0],1)
for i in range(10):
  key = raw_input("should we continue? yes/no:")
  if (key =="no"):
    print("OK, you are right.\n")
    break
  elif (key =="yes"):
    print("let's continue:\n")
    for j in range(10):
      #if (D.shape[0]==0 or D[D.shape[0]-1]<=min_D): 
      steepestAscendmethod(1)

#ax.legend()
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#plt.show()







      


