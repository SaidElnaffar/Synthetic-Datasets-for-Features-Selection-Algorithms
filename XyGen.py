# Copyright 2023 Dr. Firuz Kamalov and Dr. Said Elnaffar 

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from numpy import logical_or as lor
from numpy import logical_and as land
from numpy import logical_not as lnot
from numpy import logical_xor as lxor
import pandas as pd





'''
///////////////////////////////////////////////////////////////////////////////////////////////
A class that handles the generation of sythetic datasets for features selection algorithsm.
//////////////////////////////////////////////////////////////////////////////////////////////
'''
class XyGen():

  # You can specify the 
    def __init__(self, seed:int=0, flipping_ratio:float=0.3) -> None:
      self.seed = seed
      self.random = np.random.Generator(np.random.PCG64(self.seed) )

      self.flipping_ratio = flipping_ratio
      self.features = None
      self.y = None

    #==================================================
    # generate a list of all possible combinations of 3-bits
    def _gen_3(self): 
      rlvnt = []
      for i in [0,1]:
        for j in [0,1]:
          for k in [0,1]:
            rlvnt.append([i,j,k])
      return rlvnt


    # generate a list of all possible combinations of 3-bits
    def _gen_4(self):
      rlvnt_0 = self._gen_3()
      for seq in rlvnt_0:
        seq.append(0)

      rlvnt_1 = self._gen_3()
      for seq in rlvnt_1:
        seq.append(1)

      return rlvnt_0 + rlvnt_1


    def _r_total(self, r_array):
      r_sum = 0
      for k in range(5):
        rk_sum = 1
        for j in range(5):
          if j!=k:
            rk_sum = rk_sum*r_array[j]
        r_sum = r_sum + rk_sum
      return np.product(r_array)/r_sum


    #==================================================
    # create 2 correlated features in case of binary target (y) 
    # by randomly fliping 30% of the values of y  
    def _make_cor(self, y, flipping_ratio):
      # random.seed(0)
      cor_vars = []
      for i in range(2):
        cor_i = y.copy()
        #ind = random.sample(range(len(y)), int(flipping_ratio*len(y)))
        ind = self.random.choice(range(len(y)), int(flipping_ratio*len(y)))

        cor_i[ind] = lnot(cor_i[ind])
        cor_vars.append(cor_i)
      return np.array(cor_vars).transpose()

    #==================================================

    # make 2 correlated features when n_class => 2
    # this is more general than make_cor which works only when n_class=2
    # works by adding 1 to the y value and modding 
    # flips 30% of y values
    def _make_cor_adv(self, y, flipping_ratio, n_class=4):
      n_ind = int(flipping_ratio*len(y))
      cor_vars = []
      for i in range(2):
        #random.seed(0)
        #np.random.seed(0)

        cor_i = y.copy()
        #ind = random.sample(range(len(y)), n_ind)
        ind = self.random.choice(range(len(y)), n_ind)
        
        ##adjust = np.random.randint(n_class, size=n_ind)
        adjust = self.random.integers(n_class, size=n_ind)
        
        cor_i[ind] = (cor_i[ind]+adjust)%n_class
        cor_vars.append(cor_i)
      return np.array(cor_vars).transpose()

    #==================================================
    def loadCSV(self, csv_file:str):
      '''
      Returns features and the target by reading them from the CSV file.

              Parameters:
                      csv_file (str): file name (full path)

              Returns:
                      feature_y (tuple): a tuple of (features, y)
      '''
      assert csv_file, "Must specify a CSV file to read dataset from it."
      all = np.loadtxt(csv_file, delimiter=',')
      features, y = all[:, :-1], all[:, -1]
      return features, y

    #==================================================
    def saveCSV(self, csv_file, method:str, n_obs, relevant, cor, irrelevant, fmt="%d"):
      assert csv_file, "Must specify a CSV file name to store the dataset."
      # Save to CSV
      with open(csv_file, 'w') as f:
        #np.savetxt(csv_file, np.column_stack( (features, y) ), delimiter=',', fmt='%d', comments="#")
        f.write(f"# {'='*50}\n")
        f.write(f'# The below features have been synthesized with the following specs:\n')
        f.write(f'# Synthetic Method: {method}.\n')

        f.write(f'# Seed: {self.seed}.\n')
        f.write(f'# {n_obs } examples (rows) are generated.\n')

        f.write(f'# {len(self.features[0])} total features (columns) appear in the following order: \n')

        f.write(f'# {relevant} relevant features.\n')
        
        if cor>0:
          f.write(f'# {cor} correlated features. \n')
        
        f.write(f'# {irrelevant} irrelevant features. \n')

        f.write(f"# {'='*50}\n")

        np.savetxt(f, np.column_stack( (self.features, self.y) ), delimiter=',', fmt=fmt, comments="#")


    # ===============================================
    def gen_dataset_using_ORAND(self, n_obs=50,n_I=92, csv_file:str = None):
      red = lnot(self._gen_3()).astype(int) #redundant variables
      rr = np.hstack([self._gen_3(), red]) #rlvnt & rdnt joined
      q=n_obs//8
      r=n_obs%8
      rr_exp = np.vstack([np.repeat(rr,q, axis=0),rr[:r,:]]) #replicate rr according to n_obs
      #irlvnt = np.random.randint(2, size=[n_obs,n_I], )
      irlvnt = self.random.integers(2, size=[n_obs,n_I], )
      
      self.y = land(rr_exp[:,0], 
              lor(rr_exp[:,1], rr_exp[:,2])).astype(int) #calculate y according to the formula
      cor = self._make_cor(self.y, self.flipping_ratio)
      self.features = np.hstack([rr_exp, cor, irlvnt])
      
      # Save to file
      self.saveCSV(csv_file, "ORAND", n_obs, len(rr_exp[0]), len(cor[0]), len(irlvnt[0]))

      return self.features, self.y

    ##################################################################
    def gen_dataset_using_ِANDOR(self, n_obs=50,n_I=90, csv_file:str = None):
      red = lnot(self._gen_4()).astype(int)
      rr = np.hstack([self._gen_4(), red])
      q=n_obs//16
      r=n_obs%16
      rr_exp = np.vstack([np.repeat(rr,q, axis=0),rr[:r,:]])

      #irlvnt = np.random.randint(2, size=[n_obs,n_I])
      irlvnt = self.random.integers(2, size=[n_obs,n_I])

      self.y = lor(land(rr_exp[:,0], rr_exp[:,1]), 
              land(rr_exp[:,2], rr_exp[:,3])).astype(int)
      cor = self._make_cor(self.y, self.flipping_ratio)
      self.features = np.hstack([rr_exp, cor, irlvnt])

      # Save to file
      self.saveCSV(csv_file, "ANDOR", n_obs, len(rr_exp[0]), len(cor[0]), len(irlvnt[0]))

      return self.features, self.y

    ##################################################################
    def gen_dataset_using_ADDER(self, n_obs=50,n_I=92, csv_file:str = None):

      red = lnot(self._gen_3()).astype(int)
      rr = np.hstack([self._gen_3(), red])
      q=n_obs//8
      r=n_obs%8
      rr_exp = np.vstack([np.repeat(rr,q, axis=0),rr[:r,:]])
      #irlvnt = np.random.randint(2, size=[n_obs,n_I])
      irlvnt = self.random.integers(2, size=[n_obs,n_I])

      y1 = lxor(lxor(rr_exp[:,0], rr_exp[:,1]), 
                rr_exp[:,2]).astype(int)
      y2 = lor(land(rr_exp[:,0], rr_exp[:,1]), 
              land(rr_exp[:,2], lxor(rr_exp[:,0], rr_exp[:,1]))).astype(int)
      self.y = [y1[j] + 2*y2[j] for j in range(len(y1))]

      cor = self._make_cor_adv(np.array(self.y), self.flipping_ratio)
      self.features = np.hstack([rr_exp, cor, irlvnt])

      # Save to file
      self.saveCSV(csv_file, "ADDER", n_obs, len(rr_exp[0]), len(cor[0]), len(irlvnt[0]))

      return self.features, self.y

    ##################################################################
    def gen_dataset_using_LED(self, n_obs=180, n_I=66, df=None, config_file:str='16_segment_truth_table2.csv', csv_file:str = None):

      if not df:
        # import the table showing which LED segments light up for each character
        df = pd.read_csv(config_file, on_bad_lines='skip')
        df = df.fillna(0)
        df.index = df.iloc[:,0].values
        df.drop(columns='char', inplace=True)
        df = df.astype(int)


      rlvnt = df.values
      red = np.logical_not(rlvnt)
      rr = np.hstack([rlvnt, red])
      d = rlvnt.shape[0]
      q=n_obs//d
      r=n_obs%d
      rr_exp = np.vstack([np.repeat(rr, q, axis=0), rr[:r,:]])

      #irlvnt = np.random.randint(2, size=[n_obs,n_I])
      irlvnt = self.random.integers(2, size=[n_obs,n_I])

      y = np.array(range(36))
      self.y = np.hstack([np.repeat(y, q), y[:r]])

      cor = self._make_cor_adv(self.y, self.flipping_ratio, n_class=36)
      self.features = np.hstack([rr_exp, cor, irlvnt])


      # Save to file
      self.saveCSV(csv_file, "LED", n_obs, len(rr_exp[0]), len(cor[0]), len(irlvnt[0]))

      return self.features, self.y
    
    ##################################################################
    def gen_dataset_using_PRC(self, n_obs=50, n_I=90, csv_file:str = None):

      #rlvnt = 3 + np.random.randn(n_obs,5)/3
      rlvnt = 3 + self.random.standard_normal( (n_obs,5) )/3
      

      red = 2*rlvnt+3   #redundant features are linear transform of relevant variables
      rr = np.hstack([rlvnt, red])

      #irlvnt = 3 + np.random.randn(n_obs,n_I//2)/3
      irlvnt = 3 + self.random.standard_normal( (n_obs, n_I//2) )/3

      #irlvnt = np.hstack([irlvnt, 3+np.random.rand(n_obs,n_I//2)])
      irlvnt = np.hstack([irlvnt, 3+self.random.uniform(0, 1, size=(n_obs,n_I//2) )])
      
      
      self.features = np.hstack([rr, irlvnt])
      self.y = [self._r_total(self.features[j,:5]) for j in range(self.features.shape[0])]

      # Save to file
      self.saveCSV(csv_file, "PRC", n_obs, len(rr[0]), -1 , len(irlvnt[0]), fmt='%f')


      return self.features, self.y

      ##################################################################