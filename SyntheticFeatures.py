import numpy as np
from numpy import logical_or as lor
from numpy import logical_and as land
from numpy import logical_not as lnot
from numpy import logical_xor as lxor
import random
import pandas as pd
import matplotlib.pyplot as plt

# generate a list of all possible combinations of 3-bits
def gen_3(): 
  rlvnt = []
  for i in [0,1]:
    for j in [0,1]:
      for k in [0,1]:
        rlvnt.append([i,j,k])
  return rlvnt


    # generate a list of all possible combinations of 3-bits
def gen_4():
  rlvnt_0 = gen_3()
  for seq in rlvnt_0:
    seq.append(0)

  rlvnt_1 = gen_3()
  for seq in rlvnt_1:
    seq.append(1)

  return rlvnt_0 + rlvnt_1



# create 2 correlated features in case of binary target (y) 
# by randomly fliping 30% of the values of y  
def make_cor(y):
  random.seed(0)
  cor_vars = []
  for i in range(2):
    cor_i = y.copy()
    ind = random.sample(range(len(y)), int(0.3*len(y)))
    cor_i[ind] = lnot(cor_i[ind])
    cor_vars.append(cor_i)
  return np.array(cor_vars).transpose()



class FeaturesGenerator():
    # def __init__(self) -> None:
    #  pass
    # ===============================================
    def orand(self, n_obs=50,n_I=92, seed=0):
      np.random.seed(seed)
      red = lnot(gen_3()).astype(int) #redundant variables
      rr = np.hstack([gen_3(), red]) #rlvnt & rdnt joined
      q=n_obs//8
      r=n_obs%8
      rr_exp = np.vstack([np.repeat(rr,q, axis=0),rr[:r,:]]) #replicate rr according to n_obs
      irlvnt = np.random.randint(2, size=[n_obs,n_I], )
      y = land(rr_exp[:,0], 
              lor(rr_exp[:,1], rr_exp[:,2])).astype(int) #calculate y according to the formula
      cor = make_cor(y)
      features = np.hstack([rr_exp,cor, irlvnt])
      return features, y

    # ===============================================
    def andor(self, n_obs=50,n_I=90, seed=0):
      np.random.seed(seed)
      red = lnot(gen_4()).astype(int)
      rr = np.hstack([gen_4(), red])
      q=n_obs//16
      r=n_obs%16
      rr_exp = np.vstack([np.repeat(rr,q, axis=0),rr[:r,:]])
      irlvnt = np.random.randint(2, size=[n_obs,n_I])
      y = lor(land(rr_exp[:,0], rr_exp[:,1]), 
              land(rr_exp[:,2], rr_exp[:,3])).astype(int)
      cor = make_cor(y)
      features = np.hstack([rr_exp, cor, irlvnt])
      return features, y


#============================ Testing
obj = FeaturesGenerator()
X, y = obj.orand(seed=1)
X, y = obj.andor()

print(X, y)