import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import tqdm

class fwdODEsol(nn.Module):
    def __init__(self, theta=torch.rand(5), dt=0.1, nt=10000):
        super(fwdODEsol, self).__init__()

        self.dt   = dt
        self.nt  = nt
        self.theta = nn.Parameter(theta)
       
    def Pvfun(self, t):
      Pv  = 12.29*np.exp(-0.008605*t)
      Pvs = 10*Pv
      return Pv, Pvs
      
    def getRHS(self, P, t):
      # P = [Pint, Pb, Pintern, Pints,  Pbs,   Pinterns]
      #       B     C     D           B_c    C_c      D_c
      #   d(Sc*P)/dt = Sc*A(p)*Sc^{-1} (Sc*P)
      #  Sc = diag(1e-4, 1, 1, )    

      k1 = self.theta[0]
      k2 = self.theta[1]
      k4 = self.theta[2]
      k5 = self.theta[3]
      k6 = self.theta[4]

      Y = 2e3
      Z = 46.0
      lambda_p = 7e-5
      A = torch.zeros(6,6)
      S  = torch.zeros(6)

      
      k3   =  Z*(Y - P[1] - P[4])
      Pv, Pvs = self.Pvfun(t)
      
      # First Eq k1*A   -(k2+k3+lambda_p)*B+k4*C
      A[0,0] = -(k2+k3+lambda_p)
      A[0,1] = k4
      S[0]   = k1*Pvs

      # Second Eq  k3*B-(k4+k5+lambda_p)*C
      A[1, 0] = k3
      A[1, 1] = -(k4+k5+lambda_p)
      S[1] = 0  
      # Third Eq C*k5-(k6+lambda_p)*D
      S[2]    = 0
      A[2, 1] = k5
      A[2, 2] = -(k6+lambda_p) 

      # Forth Eq k1*A_c +  lambda_p*B-(k2+k3)*B_c+k4*C_c
      S[3]    = k1*Pv
      A[3, 0] = lambda_p
      A[3, 3] = -(k2+k3)
      A[3, 4] = k4
      
      # Fifth Eq k3*B_c+lambda_p*C-(k4+k5)*C_c;
      S[4]    = 0
      A[4, 3] = k3
      A[4, 1] = lambda_p
      A[4, 4] = -(k4+k5)

      # Fifth Eq C_c*k5+lambda_p*D-D_c*k6
      A[5, 4] = k5
      A[5, 2] = lambda_p
      A[5, 5] = -k6

      return A, S


    def forward(self, P0=torch.zeros(6)):

      dt = self.dt
      PP = torch.zeros(6, self.nt+1)
      T  = torch.zeros(self.nt+1)
      for i in range(self.nt):

        A, S = self.getRHS(PP[:,i], T[i])
        A    =  torch.eye(6) - dt*A
        R    = PP[:,i] + dt*S
        PP[:, i+1]   = torch.linalg.solve(A, R)
        T[i+1] = T[i] + dt


      return PP, T
    
# Get the data from the proposed parameters 
thetaTrue = torch.tensor([0.015, 0.016, 0.04, 1e-3, 2e-4])
forProb = fwdODEsol(thetaTrue, dt=50.0, nt=200)
Ptrue, T = forProb()
Ptrue = Ptrue.detach()

#plt.plot(T, Ptrue.t().cpu())
#plt.show()

########### Now recover paraneters
# initialization
# thetaTrue = torch.tensor([0.015, 0.016, 0.04, 1e-3, 2e-4])
thetaGuess = torch.tensor([0.010, 0.010, 0.03, 5e-3, 1e-4])
forProbC = fwdODEsol(thetaGuess, dt=50.0, nt=200)

# plot initial fit
Pinitial, T = forProbC()


optimizer = Adam(forProbC.parameters(), lr=1e-4)


# Train the network
niterations = 200

tqdm_epoch = tqdm.trange(niterations)
hh = torch.zeros(niterations)

for i in tqdm_epoch:
    
    optimizer.zero_grad()
    Pcomp, _ = forProbC()
    loss = F.mse_loss(Pcomp, Ptrue)/F.mse_loss(Ptrue*0, Ptrue) 
    
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(forProb.parameters(), 0.1)
    optimizer.step()
    
    tqdm_epoch.set_description('Average Loss: {:5e}'.format(loss))
    hh[i] = loss.item()
    
Pfinal, T = forProbC()

plt.plot(T, Ptrue.t().cpu(),'b')
plt.plot(T, Pinitial.t().detach().cpu(),'.r')
plt.plot(T, Pfinal.t().detach().cpu(),'.k')

print('Theta')

plt.show()

