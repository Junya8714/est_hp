#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
#import cvxpy
import random
import time

#-----------------------------------------------------------#
class agent:
    """エージェントのクラス"""
    def __init__(self,name,dx,dy,theta_,weight):
        self.name = name
        self.dx,self.dy = dx,dy
        self.theta = theta_
        self.weight = weight
        self.y_R = np.zeros((10,3))
        self.z_R = np.zeros((10,3))
        self.grad_K = np.zeros((len(self.theta), len(self.dx), len(self.dx)))
        self.grad_L = np.zeros(len(self.theta))
    def kernel(self,x1,x2):
        return self.theta[0]*np.exp(-(x1-x2)**2/self.theta[1]) + self.theta[2]*(x1==x2)
    def kgrad(self, xi, xj, d):
        """kの勾配"""
        if d == 0:
            return np.exp(-((xi-xj)**2)/self.theta[1])
        elif d == 1:
            return self.theta[0]*np.exp(-((xi-xj)**2)/self.theta[1])*((xi-xj)/self.theta[1])**2
        elif d == 2:
            return (xi==xj)*1.0
    def L_grad(self):
        self.K = self.kernel(*np.meshgrid(self.dx,self.dx))
        self.K_Inv = np.linalg.inv(self.K)
        """Kの勾配"""
        self.grad_K = np.zeros((len(self.theta), len(self.dx), len(self.dx)))
        for i in range(3):
            self.grad_K[i,:,:] = self.kgrad(*np.meshgrid(self.dx,self.dx), i)
        """Lの勾配"""
        self.grad_L = np.zeros(len(self.theta))
        for d in range(3):
            self.grad_L[d] = np.trace(self.K_Inv @ self.grad_K[d,:,:]) - (self.K_Inv @ self.dy).T @ self.grad_K[d,:,:] @ (self.K_Inv @ self.dy)



    #def L_solve(self):
    #    self.L = math.log(np.linalg.det(self.K)) + (self.dy).T @ (self.K_Inv @ self.dy) + len(self.dx)* math.log(2*math.pi)
    def init_y_pi(self,N):
        self.y = self.grad_L
        self.pi = (N-1)*self.y
    def Step1(self,alpha,tau,N,k):
        """近似関数を解く"""
        """self.L_solve()
        v = cvxpy.Variable(3)
        objective = cvxpy.Minimize(self.L + (self.grad_L+self.pi) @ (v-self.theta) + tau*0.5*(cvxpy.norm(v-self.theta,2))**2)
        constraints = [v>=0]
        prob = cvxpy.Problem(objective,constraints)
        result = prob.solve(solver=cvxpy.ECOS)
        self.til_theta = v.value"""
        self.til_theta = np.clip(self.theta - (self.grad_L+self.pi)/tau,1e-5,10000)
        self.z = self.theta + alpha*(self.til_theta - self.theta)
#Step2
    def receive(self,yr,zr,i):
        self.y_R[i] = yr
        self.z_R[i] = zr
    def send(self):
        return self.y,self.z
#Step3
    def Step3(self,N):
        self.y_tmp = self.grad_L
        self.theta = self.z - self.z_R[self.name,:] + self.weight @ self.z_R
        self.L_grad()
        self.y = self.grad_L - self.y_tmp + self.weight @ self.y_R
        self.pi = N * self.y - self.grad_L

#-----------------------------------------------------------#
if __name__ == '__main__':
#-----------------------------------------------------------#
    #thetaの初期値の設定

    theta = np.array(
        [[1.  , 1. ,  0.01],
         [0.6 ,0.1 ,0.013],
         [0.6 , 1.3  ,0.005],
         [1.5 , 0.4 , 0.012],
         [0.74 , 0.8 ,  0.006],
         [2. , 0.5 , 0.009],
         [1.2 , 0.71 , 0.015],
         [1.21 , 0.6  , 0.011],
         [2.   , 1. , 0.008],
         [0.7 , 0.9 , 0.0095]])
    #-----------------------------------------------------------#
    #重みの設定
    w=0.2*np.array(
      [[2,1,1,1,0,0,0,0,0,0],
       [1,2,1,0,1,0,0,0,0,0],
       [1,1,1,1,1,0,0,0,0,0],
       [1,0,1,2,0,1,0,0,0,0],
       [0,1,1,0,2,0,1,0,0,0],
       [0,0,0,1,0,2,0,1,1,0],
       [0,0,0,0,1,0,2,1,0,1],
       [0,0,0,0,0,1,1,1,1,1],
       [0,0,0,0,0,1,0,1,2,1],
       [0,0,0,0,0,0,1,1,1,2]])

    #---------------------------------------#
    iteration =1
    N=10
    agents = []
    send_k = np.zeros((6,iteration+1))
    D_List = np.zeros((6,iteration+1))
    J_List = np.zeros((6,iteration+1))

    """ガウス分布に従うようにデータセットを作成"""
    D = 1000
    np.random.seed(1)
    D_x = np.random.normal(0, 1, D)
    D_y = np.sin(np.pi*D_x)+np.random.normal(0,0.1,D)
    """用いるデータ数"""
    S = 100
    #---------------------------------------#
    for m in range(1):#閾値を変えるためのループ
        """algorithm"""

        agents.clear()
        send_number = 0
        for i in range(N):
            agents.append(agent(i,D_x[i*S:(i+1)*S],D_y[i*S:(i+1)*S],theta[i,:],w[i,:]))
        #平均
        theta_list = np.array([agents[i].theta for i in range(N)])
        mean = np.mean(theta_list,axis=0)
        #D
        for i in range(N):
            D_List[m,0] += pow(np.linalg.norm(agents[i].theta-mean,ord=2),2)
        D_List[m,0] = D_List[m,0]/N
        #J
        sum_L = np.zeros(3)
        for i in range(N):
            agents[i].theta = mean
            agents[i].L_grad()
            sum_L += agents[i].grad_L
        a = np.clip(mean - sum_L,1e-5,10000)
        J_List[m,0] = np.linalg.norm(mean - a,np.inf)
        for i in range(N):
            #agents[i].theta = theta_list[i]
            agents[i].L_grad()
            agents[i].init_y_pi(N)
        """反復"""
        for k in range(iteration):
            #alpha=1/(k+100)
            E_z = 0
            E_y = 0
            if m == 0:
                tau=10
                alpha=1/(k+1000)
                #E_z = 0
                #E_y = 0
            elif m == 1:
                alpha=1/(k+500)
                #E_z = 1000/(k+5)**5
                #E_y = 1000/(k+5)**5
            elif m == 2:
                alpha=1/(k+1000)**0.9
                #E_z = 1000/(k+5)**4
                #E_y = 1000/(k+5)**4
            elif m == 3:
                alpha=1/(k+1000)**0.9
                #E_z = 1000/(k+5)**3
                #E_y = 1000/(k+5)**3
            elif m == 4:
                tau=20
                alpha=1/(k+1000)
                #E_z = 1000/(k+5)**2
                #E_y = 1000/(k+5)**2
            elif m == 5:
                alpha=1/(k+1000)**0.9
                #E_z = 1000/(k+5)**1.5
                #E_y = 1000/(k+5)**1.5
            """step1:代替関数の最小化"""

            for i in range(N):
                agents[i].Step1(alpha,tau,N,k)

            """step2:近傍との通信"""
            for i in range(N):
                if np.linalg.norm(agents[i].z-agents[i].z_R[i],ord=2)>=E_z or np.linalg.norm(agents[i].y-agents[i].y_R[i],ord=2)>=E_y or k==0:
                    for j in range(N):
                        if w[i][j]!=0:
                            agents[j].receive(agents[i].send()[0],agents[i].send()[1],i)
                            if i != j:
                                send_number += 1
            """step3 :更新式"""
            for i in range(N):
                agents[i].Step3(N)

            """step4:評価関数"""
    #通信回数
            send_k[m,k+1] = send_number
    #平均
            theta_list = np.array([agents[i].theta for i in range(N)])
            mean = np.mean(theta_list,axis=0)
    #D
            for i in range(N):
                D_List[m,k+1] += pow(np.linalg.norm(agents[i].theta-mean,ord=2),2)
            D_List[m,k+1] = D_List[m,k+1]/N
    #J
            sum_L = np.zeros(3)
            for i in range(N):
                agents[i].theta = mean
                agents[i].L_grad()
                sum_L += agents[i].grad_L
            a = np.clip(mean - sum_L,1e-5,10000)
            J_List[m,k+1] = np.linalg.norm(mean - a,np.inf)
            for i in range(N):
                agents[i].theta = theta_list[i]
                agents[i].L_grad()
                for i in range(10):
                    print(i,agents[i].grad_L,agents[i].theta)
        print(sum_L)
    print([send_k[m,iteration] for m in range(6)])
    #経過時間表示

#plot
k_=list(range(iteration+1))
"""fig1=plt.figure()
plt.xlabel("Iteration")
plt.ylabel("D")
plt.xlim([0,iteration+1])
plt.grid(which='major', color='gray', linestyle=':')
plt.grid(which='minor', color='gray', linestyle=':')
plt.yscale('log')
plt.tight_layout()
plt.plot(k_,D_List[0,:],label='pattern1')
plt.plot(k_,D_List[1,:],label='pattern2')
plt.plot(k_,D_List[2,:],label='pattern3')
plt.plot(k_,D_List[3,:],label='pattern4')
plt.plot(k_,D_List[4,:],label='pattern5')
plt.plot(k_,D_List[5,:],label='pattern6')
plt.legend()
fig1.savefig("fig10.png")"""

fig2=plt.figure()
plt.xlabel("Iteration")
plt.ylabel("J")
plt.xlim([0,iteration+1])
plt.grid(which='major', color='gray', linestyle=':')
plt.grid(which='minor', color='gray', linestyle=':')
plt.tight_layout()
plt.yscale('log')
plt.plot(k_,J_List[0,:],label='pattern1')
plt.plot(k_,J_List[1,:],label='pattern2')
plt.plot(k_,J_List[2,:],label='pattern3')
plt.plot(k_,J_List[3,:],label='pattern4')
plt.plot(k_,J_List[4,:],label='pattern5')
plt.plot(k_,J_List[5,:],label='pattern6')
plt.legend()
fig2.savefig("fig20.png")

fig3=plt.figure()
plt.xlabel("Iteration")
plt.ylabel("communication")
plt.xlim([0,iteration+1])
plt.grid(which='major', color='gray', linestyle=':')
plt.grid(which='minor', color='gray', linestyle=':')
plt.tight_layout()
plt.plot(k_,send_k[0,:],label='pattern1')
plt.plot(k_,send_k[1,:],label='pattern2')
plt.plot(k_,send_k[2,:],label='pattern3')
plt.plot(k_,send_k[3,:],label='pattern4')
plt.plot(k_,send_k[4,:],label='pattern5')
plt.plot(k_,send_k[5,:],label='pattern6')
plt.legend()
fig3.savefig("fig30.png")
plt.show()
