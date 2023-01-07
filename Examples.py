from XyGenerator import XyGenerator

#=====================================
def test_case_1():
    '''
        Creating new datasets using different methods and saving them automatically to CSV files.
    '''
    features_generator = XyGenerator()

    X, y = features_generator.gen_dataset_using_ORAND(csv_file='orand.csv')
    print("Features: ", X)
    print("Target: ", y)



    X, y = features_generator.gen_dataset_using_ِANDOR(csv_file='andor.csv')
    print("Features: ", X)
    print("Target: ", y)


    X, y = features_generator.gen_dataset_using_ADDER(csv_file='adder.csv')
    print("Features: ", X)
    print("Target: ", y)


    url = "https://raw.githubusercontent.com/SaidElnaffar/Synthetic-Datasets-for-Features-Selection-Algorithms/0ae683f12ae664291d195aff364dbeaa2b4014d9/16_segment_truth_table2.csv"
    
    X, y = features_generator.gen_dataset_using_LED(n_obs=180,n_I=90, config_file=url, csv_file='led.csv')
    print("Features: ", X)
    print("Target: ", y)

    X, y = features_generator.gen_dataset_using_PRC(50, 90, csv_file='prc.csv')
    print("Features: ", X)
    print("Target: ", y)
    
#=====================================

def test_case_2():
    '''
        Loading a previously generated and saved dataset from a CSV file.            
    '''
    features_generator = XyGenerator()
    X, y = features_generator.loadCSV('orand.csv')
    
    print("Features: ", X)
    print("Target: ", y)

#=====================================

#************ Trying the Test Cases Above
test_case_1()

test_case_2()


#****************************************