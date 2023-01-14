from XyGen import XyGen

#=====================================
def test_case_1():
    '''
        Creating new datasets using different methods and saving them automatically to CSV files.
    '''
    path = ""

    features_generator = XyGen()

    X, y = features_generator.gen_ORAND(csv_file= path+'orand.csv')
    print('Using ORAND to generate features:')
    print("Features: ", X)
    print("Target: ", y)



    X, y = features_generator.gen_ANDOR(csv_file= path+'andor.csv')
    print('Using ANDOR to generate features:')
    print("Features: ", X)
    print("Target: ", y)


    X, y = features_generator.gen_ADDER(csv_file= path+'adder.csv')
    print('Using ADDER to generate features:')
    print("Features: ", X)
    print("Target: ", y)


    
    
    X, y = features_generator.gen_LED(n_obs=180,n_I=90, csv_file= path+'led.csv')
    print('Using LED to generate features:')
    print("Features: ", X)
    print("Target: ", y)

    X, y = features_generator.gen_PRC(50, 90, csv_file= path+'prc.csv')
    print('Using PRC to generate features:')
    print("Features: ", X)
    print("Target: ", y)
    
#=====================================

def test_case_2():
    '''
        Loading a previously generated and saved dataset from a CSV file.            
    '''
    path = ""

    features_generator = XyGen()
    X, y = features_generator.loadCSV(path+'orand.csv')
    
    print("Features: ", X)
    print("Target: ", y)

#=====================================

#************ Trying the Test Cases Above
test_case_1()

test_case_2()


#****************************************