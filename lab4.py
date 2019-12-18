import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import random

##########PART I: GENERATE COONS PATCH##########
#parameters
N = 10 #mesh size of coons patch
step = 0.1#distance between neighbor mesh
value_type = np.float64
f = np.array([],dtype=value_type) #temp: shape (-1,3) all the boundary points
alpha_u = np.array([],dtype=value_type) #blending function: shape (N,1)
beta_v = np.array([],dtype=value_type)#blending function: shape (N,1)
mash_grid = np.array([],dtype=value_type) #mesh of coons patch
surface = np.array([],dtype=value_type) #coons patch

# the following function is to determine alpha(u) and beta(v)
def buildBlendingFunction(control_point,u):
    #u is current step
    #control_point is a np.array,the shape should be (2,2)=>2 points, x-y(or called u-alpha) coordinates
    #return value is a scaler => alpha(u)
    P = np.array([0,1],dtype=value_type)
    P = np.append(P,control_point)
    P = np.append(P,[1,0])
    P = P.reshape((-1,2))
    alpha = np.array([],dtype=value_type) #shape should be (1,2)
    alpha = (1-u)**3 * P[0] + 3 * (1-u)**2 * u * P[1] + 3 * (1-u) * u**2 * P[2] + u**3 * P[3]
    return alpha[1]
    
#define coons patch
def coonsPatch(Q1,Q2,P1,P2,alpha_u,beta_v,N):
    surface_item = np.array([],dtype=value_type)
    Q00 = P1[0]
    Q01 = Q2[0]
    Q10 = P2[0]
    Q11 = P2[N]
    surface = np.array([],dtype=value_type)
    for u in range(N+1):
        alpha  = alpha_u[u]
        for v in range(N+1):
            beta = beta_v[v]
            surface_item = alpha*Q1[v]+(1-alpha)*Q2[v] + beta*P1[u]+(1-beta)*P2[u] - (beta*(alpha*Q00+(1-alpha)*Q01)+(1-beta)*(alpha*Q10+(1-alpha)*Q11))
            surface = np.append(surface,surface_item)
    surface = surface.reshape((-1,3))
    return surface

# get 4 boundary bazier curves 
def getCurve(closedOrOpen,pointsNumber,point,point_index,u):
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
    global f
    fx = 0
    fy = 0 #not this place!!
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
    f = np.append(f,array_list)

#visualize the coons patch
def visualize(alpha1,alpha2,beta1,beta2,N,show_flag):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # if show_flag == 1, show the coons patch
    #1.control point for blending function alpha(u) and beta(v)
    control_point1 = np.array([0.333333,alpha1,0.666667,alpha2],dtype=value_type)
    control_point1 = control_point1.reshape((2,2))
    control_point2 = np.array([0.333333,beta1,0.666667,beta2],dtype=value_type)
    control_point2 = control_point1.reshape((2,2))
    #2.control point for boundary curves(they should share 4 edge points!)
    point_curve1 = np.array([-10,10,10,-6,7,9,-2,7,5,2,8,9,6,11,11,10,10,10],dtype=value_type)
    point_curve1 = point_curve1.reshape((-1,3)) #P2
    point_curve2 = np.array([10,-10,10,6,6,13,9,-2,3,7,2,5,13,6,0,10,10,10],dtype=value_type)
    point_curve2 = point_curve2.reshape((-1,3)) #Q2
    point_curve3 = np.array([-10,-10,10,-6,-11,11,-2,-8,9,2,-7,5,6,-7,9,10,-10,10],dtype=value_type)
    point_curve3 = point_curve3.reshape((-1,3)) #P1
    point_curve4 = np.array([-10,-10,10,-13,3,0,-7,-2,-5,-9,2,3,-6,6,9,-10,10,10],dtype=value_type)
    point_curve4 = point_curve4.reshape((-1,3)) #Q1
    
    u = 0
    #3. get blending function
    global alpha_u,beta_v
    alpha_u = np.array([],dtype=value_type)
    beta_v = np.array([],dtype=value_type)
    for i in range(N+1):
        alpha_u_item = np.array([buildBlendingFunction(control_point1,u)],dtype=value_type)
        #print(alpha_u_item)
        alpha_u = np.append(alpha_u,alpha_u_item)
        beta_v_item = np.array([buildBlendingFunction(control_point2,u)],dtype=value_type)
        beta_v = np.append(beta_v,beta_v_item)
        u += step
    #4.get boundary curves, all the boundary points will array in f repeatedly.  
    global Q1,Q2,P1,P2,f
    Q1 = np.array([],dtype=value_type)
    Q2 = np.array([],dtype=value_type)
    P1 = np.array([],dtype=value_type)
    P2 = np.array([],dtype=value_type)
    f = np.array([],dtype=value_type)
    u = 0
    for i in range(N+1):
        getCurve(1,6,point_curve1,0,u)
        u += step
    P2 = np.append(P2,f)
    f = np.array([],dtype=value_type)
    u = 0
    for i in range(N+1):
        getCurve(1,6,point_curve2,0,u)
        u += step
    Q2 = np.append(Q2,f)
    f = np.array([],dtype=value_type)
    u = 0
    for i in range(N+1):
        getCurve(1,6,point_curve3,0,u)
        u += step
    P1 = np.append(P1,f)
    f = np.array([],dtype=value_type)
    u = 0
    for i in range(N+1):
        getCurve(1,6,point_curve4,0,u)
        u += step
    Q1 = np.append(Q1,f)
    f = np.array([],dtype=value_type)
    Q1 = Q1.reshape((-1,3))
    Q2 = Q2.reshape((-1,3))
    P1 = P1.reshape((-1,3))
    P2 = P2.reshape((-1,3))
    
    #5.show the boundary
    #plt.plot(f[:,0],f[:,1],f[:,2],'b.',markersize=1,label='open bazier curve')
    #plt.plot(point_curve1[:,0],point_curve1[:,1],point_curve1[:,2],'r.',markersize=8, label='control point1')
    #plt.plot(point_curve2[:,0],point_curve2[:,1],point_curve2[:,2],'r.',markersize=8, label='control point2')
    #plt.plot(point_curve3[:,0],point_curve3[:,1],point_curve3[:,2],'r.',markersize=8, label='control point3')
    #plt.plot(point_curve4[:,0],point_curve4[:,1],point_curve4[:,2],'r.',markersize=8, label='control point4')
    #plt.plot(point_curve1[:,0],point_curve1[:,1],point_curve1[:,2],'-',markersize=1)
    #plt.plot(point_curve2[:,0],point_curve2[:,1],point_curve2[:,2],'-',markersize=1)
    #plt.plot(point_curve3[:,0],point_curve3[:,1],point_curve3[:,2],'-',markersize=1)
    #plt.plot(point_curve4[:,0],point_curve4[:,1],point_curve4[:,2],'-',markersize=1)
    plt.plot(Q1[:,0],Q1[:,1],Q1[:,2],'r.')
    plt.plot(Q2[:,0],Q2[:,1],Q2[:,2],'r.')
    plt.plot(P1[:,0],P1[:,1],P1[:,2],'r.')
    plt.plot(P2[:,0],P2[:,1],P2[:,2],'r.')
    plt.plot(Q1[:,0],Q1[:,1],Q1[:,2],'-')
    plt.plot(Q2[:,0],Q2[:,1],Q2[:,2],'-')
    plt.plot(P1[:,0],P1[:,1],P1[:,2],'-')
    plt.plot(P2[:,0],P2[:,1],P2[:,2],'-')
    
    #6.show the surface
    global surface,mash_grid 
    surface = np.array([],dtype=value_type)
    mash_grid = np.array([],dtype=value_type)
    surface = coonsPatch(Q1,Q2,P1,P2,alpha_u,beta_v,N)
    surface = np.reshape(surface,(-1,3))
    count = N
    for i in range(count):
        for j in range(count+1):
        	mash_grid = np.append(mash_grid,surface[(count+1)*i+j])
        	mash_grid = np.append(mash_grid,surface[(count+1)*(i+1)+j])
    mash_grid = np.reshape(mash_grid,(-1,3))
    for i in range(count):
        plt.plot(surface[(count+1)*i:(count+1)*(i+1),0],surface[(count+1)*i:(count+1)*(i+1),1],surface[(count+1)*i:(count+1)*(i+1),2],'-')
        plt.plot(mash_grid[2*(count+1)*i:2*(count+1)*(i+1),0],mash_grid[2*(count+1)*i:2*(count+1)*(i+1),1],mash_grid[2*(count+1)*i:2*(count+1)*(i+1),2],'-') 
	#pass

    if show_flag == 1:
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        pass
  
#calculate the area of coons patch
def countTriangleArea(surface,N):
    area = 0
    for i in range(N):
        for j in range(N):
            x1,y1,z1 = surface[(N+1)*i+j][0],surface[(N+1)*i+j][1],surface[(N+1)*i+j][2]
            x2,y2,z2 = surface[(N+1)*(i+1)+j][0],surface[(N+1)*(i+1)+j][1],surface[(N+1)*(i+1)+j][2]
            x3,y3,z3 = surface[(N+1)*i+j+1][0],surface[(N+1)*i+j+1][1],surface[(N+1)*i+j+1][2]
            sides0 = ((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**0.5
            sides1 = ((x1 - x3)**2 + (y1 - y3)**2 + (z1 - z3)**2)**0.5
            sides2 = ((x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2)**0.5 	        
            p = (sides0 + sides1 + sides2) / 2
            area += (p * (p - sides0) * (p - sides1) * (p - sides2))**0.5
            
            x1,y1,z1 = surface[(N+1)*(i+1)+1+j][0],surface[(N+1)*(i+1)+1+j][1],surface[(N+1)*(i+1)+1+j][2] 
            sides0 = ((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**0.5
            sides1 = ((x1 - x3)**2 + (y1 - y3)**2 + (z1 - z3)**2)**0.5
            sides2 = ((x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2)**0.5 	        
            p = (sides0 + sides1 + sides2) / 2
            area += (p * (p - sides0) * (p - sides1) * (p - sides2))**0.5
    return area

##########PART II: GENETIC ALGORITHM##########
#parameters
S = [] #initial population S
S_next = [] #next population S
fitness = [] #the fitness of each individual in S
fitness_percentage = [] #the percentage of the total fitness for each s
POPULATION = 10 #the size n of the population, and POPULATION % 2 == 0
ITERATION  = 10 #the maximum running time not exceeded
S_LENGTH = 32 #length of the solution X by a binary string s, and S_LENGTH % 4 == 0

#find the decimal number for full binary number(e.g. input 3->111->7)
def fullBinToDec(length):
    sum = 0
    for i in range(length):
        sum = sum * 2 + 1
    return sum

#Decimal system->Binary system
def decToBin(num):
    arry = []   #
    while True:
        arry.append(str(num % 2))  #
        num = num // 2   
        if num == 0:    
            break
    return "".join(arry[::-1]) 

#Binary system->Decimal system
def binToDec(binary):
    result = 0   #
    for i in range(len(binary)):   
        result += int(binary[-(i + 1)]) * pow(2, i)
    return result
 
 #recover from s to alpha1,alpha2,beta1,beta2
def decoding(s,length):
     #s->each individual in S
     #length-> length of each s
     #alpha1,alpha2,beta1,beta2 belongs to [0,1]
     alpha1 =  float(binToDec(s[0:length/4])) / float(fullBinToDec(length/4))
     alpha2 =  float(binToDec(s[length/4:length/2])) / float(fullBinToDec(length/4))
     beta1 =  float(binToDec(s[length/2:length*3/4])) / float(fullBinToDec(length/4))
     beta2 =  float(binToDec(s[length*3/4:length])) / float(fullBinToDec(length/4))
     return alpha1,alpha2,beta1,beta2

#code alpha1,alpha2,beta1,beta2 to s
def coding(alpha1,alpha2,beta1,beta2,length):
     alpha1 = int(alpha1*fullBinToDec(length/4))
     alpha2 = int(alpha2*fullBinToDec(length/4)) 
     beta1 = int(beta1*fullBinToDec(length/4))
     beta2 = int(beta2*fullBinToDec(length/4))    
     return decToBin(alpha1)+decToBin(alpha2)+decToBin(beta1)+decToBin(beta2)
 
 #generate S
def generateInitialS(length,size):
    #length-> length of each s
    #size->size of the population
    global S
    for i in range(size):
        rand = random.randint(0,fullBinToDec(length))
        rand_bi = decToBin(rand)
        temp = ""
        for j in range(length-len(rand_bi)):
            temp += "0"
        rand_bi = temp + rand_bi
        S.append(rand_bi)

#Calculate the fitness of each individual in S
def  fitnessCalculator(S,length,N,show_flag):
    #population S: list
    #length-> length of each s
    #N->mesh size of coons patch
    # if show_flag == 1, show the coons patch
    fitness = []
    fitness_percentage = []
    fitness_sum = 0
    for i in range(len(S)):
        #recover from s to alpha1,alpha2,beta1,beta2
        alpha1,alpha2,beta1,beta2 = decoding(S[i],length)
        #print(S[i])
        print("alpha1:",alpha1)
        print("alpha2:",alpha2)
        print("beta1:",beta1)
        print("beta2:",beta2)
        #get area of coons patch
        visualize(alpha1,alpha2,beta1,beta2,N,show_flag)
        show_flag = 0 #only once
        #calculate the area of coons patch
        area = countTriangleArea(surface,N)
        print("area is ",area) 
        #calculate fitness
        fitness.append(1/area)
        fitness_sum += 1/area   
    for i in range(len(S)):
        fitness_percentage.append(fitness[i] / fitness_sum)
    return fitness,fitness_percentage #list

#natural selection 
def naturalSelection(S,fitness,fitness_percentage):
    #Weighted Roulette Wheel
    fitness_percentage_add = []
    S_next = []
    fitness_percentage_add.append(fitness_percentage[0])
    for i in range(len(S)-1):
        fitness_percentage_add.append(fitness_percentage[i+1]+fitness_percentage_add[i])
    for i in range(len(S)):
        rand = random.random() #0-1
        for j in range(len(S)):
            if(rand < fitness_percentage_add[j]):
                S_next.append(S[j])
                break
    return S_next

#reproduction
def reproduction(S):
    global S_next
    rand1 = random.randint(0,len(S)-1)
    rand2 = random.randint(0,len(S)-1)
    rand3 = random.randint(0,len(S[0])-1)
    s1_list = list(S[rand1])
    s2_list = list(S[rand2])
    #crossover
    for i in range(len(S[0])-rand3):
        temp = s1_list[rand3+i]
        s1_list[rand3+i] = s2_list[rand3+i]
        s2_list[rand3+i] = temp
    s1 = ""
    s2 = ""
    for i in range(len(s1_list)):
        s1 += s1_list[i]
        s2 += s2_list[i]
    S_next.append(s1)
    S_next.append(s2)

#mutation
def mutation():
    global S
    for i in range(len(S)):
        rand = random.random() #0-1
        if rand < 0.001:
            s = list(S[i])
            rand = random.randint(0,len(s)-1)
            if(s[rand] == 0):
                s[rand] = 1
            else:
                s[rand] = 0
            s1 = ""
            for i in range(len(s)):
                s1 += s[i]
            S[i] = s1

def geneticAlgorithm(N,iteration):
    global S,S_next
    show_flag = 0 # not show the figure
    for i in range(iteration):
        print("############The round of iteration is,",i,"###########")
        #calculate fitness
        fitness,fitness_percentage = fitnessCalculator(S,len(S[0]),N,show_flag)
        show_flag = 0 # not show the figure
        #natural selection 
        S = naturalSelection(S,fitness,fitness_percentage)
        #reproduction
        S_next = []
        for j in range(len(S)/2):            
            reproduction(S) 
        S = S_next
        #mutation
        mutation()
        #visualize
        if(i % 5 == 0):
            show_flag = 1 #show the figure
        print("#############################################")

##########PART III: main()##########
#main() for PART I 
#visualize(0.33333,0.66666,0.33333,0.66666,N)
#ax.legend()
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#plt.show()

#main() for PART II
generateInitialS(S_LENGTH,POPULATION)
geneticAlgorithm(N,ITERATION)










