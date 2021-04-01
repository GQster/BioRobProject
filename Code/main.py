
#run this line:
#pyenv version
#if not python 3.7.9
#pyenv install 3.7.9
import numpy as np
import pandas as pd
import xlrd						        # So can read the spreadsheet IF IS A .XLS FILE
from IPython.display import display     #for display

#from hello import *             #imports entire file
#from hello import my_function  #imports sepcific function


#workbook = xlrd.open_workbook('Monthly_Data_Input.xls')	#opens the spreadsheet in python
#worksheetINFO = workbook.sheet_by_index(0)			        #opens the first worksheet in the spreadsheet


experiment_data_1066528_filtered = pd.read_csv('Data/1066528_heartrate_filtered.csv')

display(experiment_data_1066528_filtered)

print(experiment_data_1066528_filtered[0])
print('applesauce')
print('applesauce2')