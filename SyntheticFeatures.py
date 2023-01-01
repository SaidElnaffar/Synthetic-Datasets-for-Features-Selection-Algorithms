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
def make_cor(y, flipping_ratio):
  random.seed(0)
  cor_vars = []
  for i in range(2):
    cor_i = y.copy()
    ind = random.sample(range(len(y)), int(flipping_ratio*len(y)))
    cor_i[ind] = lnot(cor_i[ind])
    cor_vars.append(cor_i)
  return np.array(cor_vars).transpose()


# make 2 correlated features when n_class => 2
# this is more general than make_cor which works only when n_class=2
# works by adding 1 to the y value and modding 
# flips 30% of y values

def make_cor_adv(y, flipping_ratio, n_class=4):
  n_ind = int(flipping_ratio*len(y))
  cor_vars = []
  for i in range(2):
    random.seed(0)
    np.random.seed(0)
    cor_i = y.copy()
    ind = random.sample(range(len(y)), n_ind)
    adjust = np.random.randint(n_class, size=n_ind)
    cor_i[ind] = (cor_i[ind]+adjust)%n_class
    cor_vars.append(cor_i)
  return np.array(cor_vars).transpose()



def read_df():
  # import the table showing which LED segments light up for each character
  df = pd.read_csv('16_segment_truth_table2.csv')
  df = df.fillna(0)
  df.index = df.iloc[:,0].values
  df.drop(columns='char', inplace=True)
  df = df.astype(int)
  return df



def r_total(r_array):
  r_sum = 0
  for k in range(5):
    rk_sum = 1
    for j in range(5):
      if j!=k:
        rk_sum = rk_sum*r_array[j]
    r_sum = r_sum + rk_sum
  return np.product(r_array)/r_sum






class FeaturesGenerator():
    def __init__(self, seed:float=0, flipping_ratio:float=0.3) -> None:
      self.seed = seed
      self.flipping_ratio = flipping_ratio
      self.features = None
      self.y = None

    #==================================================
    def load(self, csv_file):
      assert csv_file
      all = np.loadtxt(csv_file, delimiter=',', dtype=int)
      features, y = all[:, :-1], all[:, -1]
      return features, y

    #==================================================
    def saveCSV(self, csv_file, method:str, seed, n_obs, relevant, cor, irrelevant, fmt="%d"):
      assert csv_file
      # Save to CSV
      with open(csv_file, 'w') as f:
        #np.savetxt(csv_file, np.column_stack( (features, y) ), delimiter=',', fmt='%d', comments="#")
        f.write(f"# {'='*50}\n")
        f.write(f'# The below features have been synthesized with the following specs:\n')
        f.write(f'# Synthetic Method: {method}.\n')

        f.write(f'# Seed: {seed}.\n')
        f.write(f'# {n_obs } examples (rows) are generated.\n')

        f.write(f'# {len(self.features[0])} total features (columns) appear in the following order: \n')

        f.write(f'# {relevant} relevant features.\n')
        
        if cor>0:
          f.write(f'# {cor} correlated features. \n')
        
        f.write(f'# {irrelevant} irrelevant features. \n')

        f.write(f"# {'='*50}\n")

        np.savetxt(f, np.column_stack( (self.features, self.y) ), delimiter=',', fmt=fmt, comments="#")


    # ===============================================
    def orand(self, n_obs=50,n_I=92, seed=None, csv_file:str = None):
      # If no seed is sent, then use the default one of the generator instance
      seed = self.seed if not seed else seed

      np.random.seed(seed)
      red = lnot(gen_3()).astype(int) #redundant variables
      rr = np.hstack([gen_3(), red]) #rlvnt & rdnt joined
      q=n_obs//8
      r=n_obs%8
      rr_exp = np.vstack([np.repeat(rr,q, axis=0),rr[:r,:]]) #replicate rr according to n_obs
      irlvnt = np.random.randint(2, size=[n_obs,n_I], )
      
      self.y = land(rr_exp[:,0], 
              lor(rr_exp[:,1], rr_exp[:,2])).astype(int) #calculate y according to the formula
      cor = make_cor(self.y, self.flipping_ratio)
      self.features = np.hstack([rr_exp, cor, irlvnt])
      
      # Save to file
      self.saveCSV(csv_file, "ORAND", seed, n_obs, len(rr_exp[0]), len(cor[0]), len(irlvnt[0]))

      return self.features, self.y

    ##################################################################
    def andor(self, n_obs=50,n_I=90, seed=None, csv_file:str = None):
      # If no seed is sent, then use the default one of the generator instance
      seed = self.seed if not seed else seed
      
      
      np.random.seed(seed)
      red = lnot(gen_4()).astype(int)
      rr = np.hstack([gen_4(), red])
      q=n_obs//16
      r=n_obs%16
      rr_exp = np.vstack([np.repeat(rr,q, axis=0),rr[:r,:]])
      irlvnt = np.random.randint(2, size=[n_obs,n_I])
      self.y = lor(land(rr_exp[:,0], rr_exp[:,1]), 
              land(rr_exp[:,2], rr_exp[:,3])).astype(int)
      cor = make_cor(self.y, self.flipping_ratio)
      self.features = np.hstack([rr_exp, cor, irlvnt])

      # Save to file
      self.saveCSV(csv_file, "ANDOR", seed, n_obs, len(rr_exp[0]), len(cor[0]), len(irlvnt[0]))

      return self.features, self.y


    ##################################################################
    def adder(self, n_obs=50,n_I=92, seed=None, csv_file:str = None):

      # If no seed is sent, then use the default one of the generator instance
      seed = self.seed if not seed else seed

      np.random.seed(seed)
      red = lnot(gen_3()).astype(int)
      rr = np.hstack([gen_3(), red])
      q=n_obs//8
      r=n_obs%8
      rr_exp = np.vstack([np.repeat(rr,q, axis=0),rr[:r,:]])
      irlvnt = np.random.randint(2, size=[n_obs,n_I])
      y1 = lxor(lxor(rr_exp[:,0], rr_exp[:,1]), 
                rr_exp[:,2]).astype(int)
      y2 = lor(land(rr_exp[:,0], rr_exp[:,1]), 
              land(rr_exp[:,2], lxor(rr_exp[:,0], rr_exp[:,1]))).astype(int)
      self.y = [y1[j] + 2*y2[j] for j in range(len(y1))]

      cor = make_cor_adv(np.array(self.y), self.flipping_ratio)
      self.features = np.hstack([rr_exp, cor, irlvnt])

      # Save to file
      self.saveCSV(csv_file, "ADDER", seed, n_obs, len(rr_exp[0]), len(cor[0]), len(irlvnt[0]))

      return self.features, self.y

    ##################################################################
    def led(self, n_obs=180, n_I=66, seed=None, df=None, csv_file:str = None):
      # If no seed is sent, then use the default one of the generator instance
      seed = self.seed if not seed else seed
      np.random.seed(seed)

      if not df:
        df = read_df()

      rlvnt = df.values
      red = np.logical_not(rlvnt)
      rr = np.hstack([rlvnt, red])
      d = rlvnt.shape[0]
      q=n_obs//d
      r=n_obs%d
      rr_exp = np.vstack([np.repeat(rr, q, axis=0), rr[:r,:]])
      irlvnt = np.random.randint(2, size=[n_obs,n_I])
      y = np.array(range(36))
      self.y = np.hstack([np.repeat(y, q), y[:r]])

      cor = make_cor_adv(self.y, self.flipping_ratio, n_class=36)
      self.features = np.hstack([rr_exp, cor, irlvnt])


      # Save to file
      self.saveCSV(csv_file, "LED", seed, n_obs, len(rr_exp[0]), len(cor[0]), len(irlvnt[0]))

      return self.features, self.y
    
    ##################################################################
    def prc(self, n_obs=50, n_I=90, seed=None, csv_file:str = None):
      # If no seed is sent, then use the default one of the generator instance
      seed = self.seed if not seed else seed

      np.random.seed(seed)
      rlvnt = 3 + np.random.randn(n_obs,5)/3
      red = 2*rlvnt+3   #redundant features are linear transform of relevant variables
      rr = np.hstack([rlvnt, red])

      irlvnt = 3 + np.random.randn(n_obs,n_I//2)/3
      irlvnt = np.hstack([irlvnt, 3+np.random.rand(n_obs,n_I//2)])
      
      self.features = np.hstack([rr, irlvnt])
      self.y = [r_total(self.features[j,:5]) for j in range(self.features.shape[0])]

      # Save to file
      self.saveCSV(csv_file, "PRC", seed, n_obs, len(rr[0]), -1 , len(irlvnt[0]), fmt='%f')


      return self.features, self.y

      ##################################################################