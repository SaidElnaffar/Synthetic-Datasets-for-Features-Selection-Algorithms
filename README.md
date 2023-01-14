
# XyGen: ML Features Generator

XyGen is a machine learning library that assists researchers in generating synthetic datasets for evaluating feature selection algorithms. The X refers to the commonly known dependent features and the y refers to the dependent target attribute. The library comprises a single, self-contained module and currently includes 5 different methods for generating artificial datasets: ORAND, ANDOR, ADDER, LED, and PRC. These methods are primarily based on concepts in computer science and electronics. Additionally, XyGen is flexible and can easily be extended to include other custom generation methods.

To use the XyGen module, you simply import the XyGen class from the XyGen module and instantiate a generator object from that class. Using this generator object you can generate features that can be used for benchmarking a suite of features selection algorithms. The synthesized datasets can be saved and loaded from CSV.

## Usage/Examples

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