from SyntheticFeatures import FeaturesGenerator

#============================ Testing
generator = FeaturesGenerator()
X, y = generator.orand(seed=1)
X, y = generator.andor()
X, y = generator.adder(n_obs=50,n_I=92, seed=0)
X, y = generator.led(180, 90)
X, y = generator.prc(50, 90, 0)
print(X, y)