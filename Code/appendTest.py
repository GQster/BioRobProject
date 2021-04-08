from functions import *     #imports our functions
#from main import *          #imports HR_list and Sleep_list


# A place for me to code without messing up  stuff
#X = get_data('Sleep_HR_0.csv')
X = pd.DataFrame([], columns= ['SleepLVL', 'HR'])
for filex in Sleep_HR_list:
    Xtemp = get_data(filex)                    # Returns X; holding sleep states and HR data    
    X.append(Xtemp)

display(X)

