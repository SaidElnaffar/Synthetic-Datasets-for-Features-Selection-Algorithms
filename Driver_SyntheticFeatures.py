from SyntheticFeatures import FeaturesGenerator

#=====================================
def test_case_1():
    '''
        Loading a previously generated and saved dataset from a CSV file.            
    '''
    features_generator = FeaturesGenerator()
    X, y = features_generator.loadCSV()
    
    print("Features: ", X)
    print("Target: ", y)
#=====================================
def test_case_2():
    '''
        Creating new datasets using different methods and saving them automatically to CSV files.
    '''
    features_generator = FeaturesGenerator()

    X, y = features_generator.gen_dataset_using_ORAND(seed=1, csv_file='orand.csv')
    print("Features: ", X)
    print("Target: ", y)



    X, y = features_generator.gen_dataset_using_ِANDOR(csv_file='andor.csv')
    print("Features: ", X)
    print("Target: ", y)


    X, y = features_generator.gen_dataset_using_ADDER(csv_file='adder.csv')
    print("Features: ", X)
    print("Target: ", y)

    X, y = features_generator.gen_dataset_using_LED(n_obs=180,n_I=90, csv_file='led.csv')
    print("Features: ", X)
    print("Target: ", y)

    X, y = features_generator.gen_dataset_using_PRC(50, 90, csv_file='prc.csv')
    print("Features: ", X)
    print("Target: ", y)
    
#=====================================

#************ Trying the Test Cases Above
test_case_1()

test_case_2()


#****************************************