from SyntheticFeatures import FeaturesGenerator

#============================ Testing
generator = FeaturesGenerator()
X, y = generator.orand(seed=1, csv_file='orand.csv')
X, y = generator.andor(csv_file='andor.csv')
X, y = generator.adder(csv_file='adder.csv')
X, y = generator.led(n_obs=180,n_I=90, csv_file='led.csv')
X, y = generator.prc(50, 90, csv_file='prc.csv')


'''
X, y = generator.andor()
X, y = generator.adder(n_obs=50,n_I=92, seed=0)
X, y = generator.led(180, 90)
X, y = generator.prc(50, 90, 0)
'''

# print(X)
# print('Target: ', y)
print('********************************************')
#all = generator.load('orand.csv')
#all = generator.load('andor.csv')
#X, y  = generator.load('prc.csv')
print( X, y)

