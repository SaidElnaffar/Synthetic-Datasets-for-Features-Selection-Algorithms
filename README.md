
# XyGen: ML Features Generator

XyGen is a machine learning library that assists researchers in generating synthetic datasets for evaluating feature selection algorithms. The X refers to the commonly known dependent features and the y refers to the dependent target attribute. The library comprises a single, self-contained module and currently includes 5 different methods for generating artificial datasets: ORAND, ANDOR, ADDER, LED, and PRC. These methods are primarily based on concepts in computer science and electronics. Additionally, XyGen is flexible and can easily be extended to include other custom generation methods.

To use the XyGen module, you simply import the XyGen class from the XyGen module and instantiate a generator object from that class. Using this generator object you can generate features that can be used for benchmarking a suite of features selection algorithms. The synthesized datasets can be saved and loaded from CSV.

## Usage/Examples

from XyGen import XyGen


def test_case_1():

    '''
        Creating new datasets using different methods and saving them automatically to CSV files.
    '''
    features_generator = XyGen()

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
    features_generator = XyGen()
    X, y = features_generator.loadCSV('orand.csv')
    
    print("Features: ", X)
    print("Target: ", y)

#=====================================


