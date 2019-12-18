import numpy as np
import matplotlib.pyplot as plt
import math

N = 10
step = 0.1
value_type = np.float64

####1.define boundary curve####
S = 8.
R = 5.
gap = 1.5
Pi = 3.1415926
count = 10
count_rev = 0.1
#Q1
Q1 = np.array([],dtype=value_type)
arc = 2 * (R**2 - (R-gap)**2)**0.5
#Q2
Q2 = np.array([],dtype=value_type)
angleQ2 = math.acos(0.5*S / R)
#P1
P1 = np.array([],dtype=value_type)
angleP1 = math.acos((R-gap) / R)
#P2
P2 = np.array([],dtype=value_type)

for i in range(N+1):
  #Q1
  temp = np.array([-(R-gap),-0.5*arc+i*step*arc],dtype=value_type)
  Q1 = np.append(Q1,temp)
  #Q2
  temp = np.array([R*math.cos(Pi+angleQ2-i*step*2*angleQ2)+S,R*math.sin(Pi+angleQ2-i*step*2*angleQ2)],dtype=value_type)
  Q2 = np.append(Q2,temp)
  #P1
  temp = np.array([R*math.cos(Pi+angleP1+i*step*(Pi-angleP1-angleQ2)),R*math.sin(Pi+angleP1+i*step*(Pi-angleP1-angleQ2))],dtype=value_type)
  P1 = np.append(P1,temp)
  #P2
  temp = np.array([R*math.cos(Pi-angleP1-i*step*(Pi-angleP1-angleQ2)),R*math.sin(Pi-angleP1-i*step*(Pi-angleP1-angleQ2))],dtype=value_type)
  P2 = np.append(P2,temp)

Q1 = Q1.reshape((-1,2))
Q2 = Q2.reshape((-1,2))
P1 = P1.reshape((-1,2))
P2 = P2.reshape((-1,2))

####2.define origin mesh#### this part is abondened
mesh = np.array([],dtype=value_type)
#row 0-99 * 0-10 11
for i in range(count+1):
  for j in range(N):
    temp = np.array([i*count_rev,j*step],dtype=value_type)
    mesh = np.append(mesh,temp)
#col 0-99 * 11-21 11
for i in range(count+1):
  for j in range(N):
    temp = np.array([j*step,i*count_rev],dtype=value_type)
    mesh = np.append(mesh,temp)
#screw 0-99 * 22-31 10
for i in range(count):
  for j in range(N):
    l = (i+1)*count_rev
    temp = np.array([l-l*step*j,l*step*j],dtype=value_type)
    mesh = np.append(mesh,temp)
#screw 0-99 * 32-41 10
for i in range(count):
  for j in range(N):
    l = 1-(i+1)*count_rev
    temp = np.array([1-l*step*j,(i+1)*count_rev+l*step*j],dtype=value_type)
    mesh = np.append(mesh,temp)

mesh = mesh.reshape((-1,2))

####3.define blending function####
control_point = np.array([],dtype=value_type)#shape 3,2
def buildBlendingFunction(control_point,u,switch):
  #u is current step
  #control_point is a np.array,the shape should be (2,2)=>2 points, x-y(or called u-alpha) coordinates
  #return value is a scaler => alpha(u)
  #switch is to decide to make alpha the bezier one or normal one
  if(switch==1):
    P = np.array([0,1],dtype=value_type)
    P = np.append(P,control_point)
    P = np.append(P,[1,0])
    P = P.reshape((-1,2))
    alpha = np.array([],dtype=value_type) #shape should be (1,2)
    alpha = (1-u)**3 * P[0] + 3 * (1-u)**2 * u * P[1] + 3 * (1-u) * u**2 * P[2] + u**3 * P[3]
    #print("alpha\n",alpha)
    #print("alpha\n",alpha[0],alpha[1])
    #print(P[0],P[1])
    #plt.scatter(alpha[0],alpha[1],markersize=1)
  if(switch==2):
    alpha = np.array([0,0],dtype=value_type) #shape should be (1,2)
    alpha[1] = 1 - u*step
  return alpha[1]

####4.define coons patch####
def coonsPatch(Q1,Q2,P1,P2):
  surface_item = np.array([],dtype=np.float64)
  Q00 = P1[0]
  Q01 = Q2[0]
  Q10 = P2[0]
  Q11 = P2[count]
  surface = np.array([],dtype=np.float64)
  for u in range(count+1):
    alpha_u = buildBlendingFunction(control_point,u,2)
    for v in range(count+1):
      beta_v = buildBlendingFunction(control_point,v,2)
      surface_item = alpha_u*Q1[v]+(1-alpha_u)*Q2[v] + beta_v*P1[u]+(1-beta_v)*P2[u] - (beta_v*(alpha_u*Q00+(1-alpha_u)*Q01)+(1-beta_v)*(alpha_u*Q10+(1-alpha_u)*Q11))
      surface = np.append(surface,surface_item)
  surface = surface.reshape((-1,2))
  return surface

surface = coonsPatch(Q1,Q2,P1,P2)

####4.mapping####
mash_grid = np.array([],dtype=np.float64)
for i in range(count):
  for j in range(count+1):
    mash_grid = np.append(mash_grid,surface[(count+1)*i+j])
    mash_grid = np.append(mash_grid,surface[(count+1)*(i+1)+j])
mash_grid = np.reshape(mash_grid,(-1,2))
for i in range(count):
  plt.plot(surface[(count+1)*i:(count+1)*(i+1),0],surface[(count+1)*i:(count+1)*(i+1),1])
  plt.plot(mash_grid[2*(count+1)*i:2*(count+1)*(i+1),0],mash_grid[2*(count+1)*i:2*(count+1)*(i+1),1])   

#plt.subplot(121)
plt.plot(Q1[:,0],Q1[:,1])
plt.plot(Q2[:,0],Q2[:,1])
plt.plot(P1[:,0],P1[:,1])
plt.plot(P2[:,0],P2[:,1])
#plt.plot(surface[:,0],surface[:,1])
plt.xlim(-6,6)
plt.ylim(-6,6)

#plt.subplot(122)
#for i in range(mesh.shape[0]/100):
#  plt.plot(mesh[(count*i):(count+1)*i-1,0],mesh[count*i:(count+1)*i-1,1])
#plt.plot(mesh[:,0],mesh[:,1])
plt.show()

####5.calculate area####
#S=(x1y2-x1y3+x2y3-x2y1+x3y1-x2y2) which is 3 points of a triangle
area = 0.
for i in range(count):
  for j in range(count):
    x1 = surface[11*i+j][0]
    y1 = surface[11*i+j][1]
    x2 = surface[11*i+11+j][0]
    y2 = surface[11*i+11+j][1]
    x3 = surface[11*i+1+j][0]
    y3 = surface[11*i+1+j][1]    
    area += abs((x1*y2+x2*y3+x3*y1-x1*y3-x2*y1-x3*y2)/2)#(x1*y2-x1*y3+x2*y3-x2*y1+x3*y1-x2*y2)
    x1 = surface[11*i+12+j][0] 
    y1 = surface[11*i+12+j][1]    
    area += abs((x1*y2+x2*y3+x3*y1-x1*y3-x2*y1-x3*y2)/2)# (x1*y2-x1*y3+x2*y3-x2*y1+x3*y1-x2*y2)
print("area is:",area)   
   
####6.Laplacian####
def Laplacian(surface,iteration):
  surface2 = np.array([],dtype=np.float64)
  surface2 = np.append(surface,surface2) 
  surface2 = surface2.reshape((-1,2))     
  for k in range(iteration):
    for i in range(count-1):
      for j in range(count-1):
        surface2[(i+1)*11+j+1] = ((surface2[(i+1)*11+j] + surface2[(i+1)*11+j+2] + surface2[i*11+j+1] + surface2[(i+2)*11+j+1] + surface2[i*11+j+2] + surface2[(i+2)*11+j]) /6 + surface2[(i+1)*11+j+1])/2
  surface2 = np.reshape(surface2, (-1,2))
  mash_grid2 = np.array([],dtype=np.float64)
  for i in range(count):
    for j in range(count+1):
      mash_grid2 = np.append(mash_grid2,surface2[11*i+j])
      mash_grid2 = np.append(mash_grid2,surface2[11*(i+1)+j])
  mash_grid2 = mash_grid2.reshape((-1,2)) 
  return surface2,mash_grid2

#fig = plt.figure(figsize = (18,6))
for i in range(3):
  surface2,mash_grid2 = Laplacian(surface,(2*i**2+2)*10)
  #ax = fig.add_subplot(1,3,i+1)
  plt.title("Iteration : {}".format((2*i**2+2)*10))
  #ax.grid()
  plt.xlim(-6 , 6)
  plt.ylim(-6 , 6)
    
  for i in range(count):
    smooth1 = plt.plot(surface2[11*i:11+11*i,0],surface2[11*i:11+11*i,1], c='r')
    smooth2 = plt.plot(mash_grid2[22*i:22+22*i,0],mash_grid2[22*i:22+22*i,1], c='r')
  for i in range(count):
      origin1 = plt.plot(surface[11*i:11+11*i,0],surface[11*i:11+11*i,1], c='g')
      origin2 = plt.plot(mash_grid[22*i:22+22*i,0],mash_grid[22*i:22+22*i,1], c='g')
  #ax.plot(Q1[:,0],Q1[:,1], c='blue')
  #ax.plot(Q2[:,0],Q2[:,1], c='blue')
  #ax.plot(P1[:,0],P1[:,1], c='blue')
  #ax.plot(P2[:,0],P2[:,1], c='blue')
  plt.plot(Q1[:,0],Q1[:,1], c='blue')
  plt.plot(Q2[:,0],Q2[:,1], c='blue')
  plt.plot(P1[:,0],P1[:,1], c='blue')
  plt.plot(P2[:,0],P2[:,1], c='blue')
    #ax.legend((origin1,smooth1),('Before','after'))
  plt.show()
    
   








