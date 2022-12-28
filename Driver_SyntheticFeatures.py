from SyntheticFeatures import FeaturesGenerator

#============================ Testing
generator = FeaturesGenerator()
X, y = generator.orand(seed=1, csv_file='orand.csv')

'''
X, y = generator.andor()
X, y = generator.adder(n_obs=50,n_I=92, seed=0)
X, y = generator.led(180, 90)
X, y = generator.prc(50, 90, 0)
'''

print(X)
print('Target: ', y)
print('********************************************')
all = generator.load('orand.csv')
print( all[0], all[1])

