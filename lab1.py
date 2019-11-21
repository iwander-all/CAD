########################################
# 11/5/2019
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

    #database = np.loadtxt(filepath,delimiter=' ')
    #database = pd.read_csv(filepath,sep=' ')
    #print(database)
    #m = database[0,:] #the number of Beziercurve
    #u = 0 # index for database

    for i in range(m[0]): 
    #  u += 1
      pointsNumber_str = data.readline()
      pointsNumber_list = list(pointsNumber_str.split())
      temp1 = np.array(pointsNumber_list,dtype=np.uint8)
      global pointsNumber
      pointsNumber = np.append(pointsNumber,temp1) # the number of control points of the ith Bezier curve
    #u += 1

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
  
  point = np.reshape(point, (-1,3))
  print(m)
  print(pointsNumber)
  print(closedOrOpen)
  print(point)

#conclude ===>

#1.when input, if it is more than one word, you must use "raw_input" instead of "input"(very easily to make mistakes!!!)
#2.if you want from string, to array, you must string => list => array
#3.readline is the best way when the number is not the same in each row instead of np.loadtxt()
#4. when you use np.tostring(), "2" will not be 2 but 50! "2 " will be two number like [50 10]
#5.use "for i in range(m[0])" instead of "for i in range(m)" or something else!
#6. use "global" if you want to a global variaty in a function, or it will not change once the function determinated.


def inputFromKeyboard(): 
  
  m_int = input("the number of Beziercurve:") #the number of Beziercurve
  global m
  m = np.array([m_int],dtype = np.uint8)

  filename = raw_input("please input filename:(example: data.txt):")  #input can only read 1 word!
  
  with open(filename,mode="w") as file_txt:
    file_txt.write(str(m_int)+'\n') #change to string!!

    for i in range(m_int): 
      pointsNumber_str = str(input("numbers of control points of the {}th Bezier curve:".format(i)))
      pointsNumber_list = list(pointsNumber_str.split())
      temp1 = np.array(pointsNumber_list,dtype=np.uint8)
      global pointsNumber
      pointsNumber = np.append(pointsNumber,temp1) # the number of control points of the ith Bezier curve
    #u += 1
      file_txt.write(pointsNumber_str + '\n')

      closedOrOpen_str = str(input("0 or 1: 0 means the ith curve is closed, 1 means it is open:")) #0 means the ith curve is closed, 1 means it is open
      closedOrOpen_list = list(closedOrOpen_str.split())
      temp2 = np.array(closedOrOpen_list,dtype=np.uint8)
      global closedOrOpen
      closedOrOpen = np.append(closedOrOpen,temp2) #0 means the ith curve is closed, 1 means it is open
      file_txt.write(closedOrOpen_str + '\n')

      global point
      for j in range(temp1[0]):
        point_str = str(raw_input("the coordinates of control points for the {}th Bezier Curve (eg: '0 0 0'):".format(i))) # control points for the ith Bezier Curve
        point_list = list(point_str.split())
        temp3 = np.array(point_list,dtype=np.float64)
        point = np.append(point,temp3) # control points for the ith Bezier Curve
        file_txt.write(point_str + '\n')

  point = np.reshape(point, (-1,3))
  print(m)
  print(pointsNumber)
  print(closedOrOpen)
  print(point)


def drawCurve(closedOrOpen,pointsNumber,point,point_index):

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
      #C[i] = up / down
      C.append(up / down)
      #C[n-i] = C[i]
  elif ((n+1) % 2 == 1):
    for i in range(n / 2):
      up = 1
      down = 1
      j = n
      while (j > i):
        up *= j
        #print(up)
        j = j - 1
      j = n - i
      while (j > 0):
        down *= j
        #print(down)
        j = j - 1
      #C[i] = up / down
      C.append(up / down)
      #C[n-i] = C[i]
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
    #C[n/2] = up/down
    C.append(up / down)
  if (n%2 == 1):
    for i in range(int((n+1)/2)):
      C.append(C[int(n/2-i)])
  if (n%2 == 0):
    for i in range(int((n+1)/2)):
      C.append(C[int(n/2-i-1)])
  print("C",C) 
  #C = np.array(C,dtype=np.float64)
  #u = np.arange(0,1,0.01)
  f = np.array([],dtype=np.float64)
  #print(f.shape)
  #f.reshape(-1,3)
  #print(f.shape)
  #print(point[point_index].shape)
  #print(point.shape)
  #print(point[point_index][0])
  #print(point[point_index][1])
  #print(point[point_index][2])
  #print(point)
  #print(len(C))
  #print(n)

  #f[1][0] += C[i] * np.power(u,1) * np.power((1-u),(n-1)) * point[point_index + 1][0]
  
  u = 0

  #f_unit = np.array([[0,0,0]],dtype=np.float64)
  #f_unit.reshape(1,3) 

  fx = 0
  fy = 0 #not this place!!
  fz = 0
  #print(f_unit.shape)
  print("point_show\n",point_show)
  print("n+1",n+1)
  for j in range(100): 
    #f[1][0] += C[i] * np.power(u,1) * np.power((1-u),(n-1)) * point[point_index + 1][0]
    #print(f.shape)
    fx = 0
    fy = 0  #do not forget return 0!!!!!!!!!!!!
    fz = 0
    #if u == 0:
    #  fx += C[i] * point_show[0][0]
    #  fy += C[i] * point_show[0][1]
    #  fz += C[i] * point_show[0][2]
    #elif u == 1:
    #  fx += C[i] * point_show[n][0]
    #  fy += C[i] * point_show[n][1]
    #  fz += C[i] * point_show[n][2]
    #else:
    for i in range(n+1):
      fx += C[i] * u**i * (1-u)**(n-i) * point_show[i][0]
      fy += C[i] * u**i * (1-u)**(n-i) * point_show[i][1]
      fz += C[i] * u**i * (1-u)**(n-i) * point_show[i][2]
    list = []
    list.append(fx)
    list.append(fy) 
    list.append(fz)
    array_list = np.array([list],dtype=np.float64) 
    u += 0.01
    f = np.append(f,array_list)
  f_show = f.reshape((-1,3))
  print(f_show.shape)
  #print(f_show)

  #show the control lines!
  #control_line_show = np.array([],dtype=np.float64)
  #for u in range(100): 
  #  for i in range(n):
  #    temp0 =temp1=temp2=0
  #    temp0 = u*point_show[i][0]+(1-u)*point_show[i+1][0]
  #    temp1 = u*point_show[i][1]+(1-u)*point_show[i+1][1]
  #    temp2 = u*point_show[i][2]+(1-u)*point_show[i+1][2]
  #    temp = np.array([temp0,temp1,temp2],dtype=np.float64)
  #    control_line_show = np.append(control_line_show, temp)
  #control_line_show = control_line_show.reshape((-1,3))

  #ax.scatter(control_line_show[:,0],control_line_show[:,1],control_line_show[:,2])#,'y.',markersize=1, label='control line')

  #fig = plt.figure()
  #ax = fig.gca(projection='3d')
  ax.plot(f_show[:,0],f_show[:,1],f_show[:,2],'b.',markersize=1,label='open bazier curve')
  ax.plot(point_show[:,0],point_show[:,1],point_show[:,2],'-',markersize=1, label='control line')
  ax.plot(point_show[:,0],point_show[:,1],point_show[:,2],'r.',markersize=8, label='control points')
  print(point_show.shape[0])

  #for i in range(point_show.shape[0]):
  #  plt.plot(point_show[i,0],point_show[i,1],point_show[i,2],'y.', label='control point')
  #ax.legend()
  #ax.set_xlabel('X')
  #ax.set_ylabel('Y')
  #ax.set_zlabel('Z')
  #plt.show()


#conclusion ==>
#1.f.reshape((-1,3)) is wrong!! it should be f_show = f.reshape((-1,3)) !
#2.after append something, the shape of the array will change!
#3.If you want to add something to array or list, we cannot use C[i] = ww, but use append!


# main() ==>

inputType = input("Please choose data from text/0 or keyboard/1 ?:")
#print(inputType.type())

if ( inputType == 0): # not '0'!!
  inputFromText()
elif ( inputType == 1):
  inputFromKeyboard()

point_index = 0
fig = plt.figure()
ax = fig.gca(projection='3d')

for i in range(m[0]):
  drawCurve(closedOrOpen[i],pointsNumber[i],point,point_index)
  point_index += pointsNumber[i]

ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
  #else:
  #  print("Error: Invalid index!")


