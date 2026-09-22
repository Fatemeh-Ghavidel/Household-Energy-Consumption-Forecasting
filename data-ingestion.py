
#################################### DATA INGESTION ####################################
#### DATA LOADING 
#### DATA OVERALL VIEW; Data type, Memory usage, Summary of statical information, Missing values
#### NAN VALUES VISUALISATION




#===========================================================================================
        #%% Data Loading
df = pd.read_csv(r"D:\Machine learning\EnergyProject\New folder (2)\household_power_consumption.txt", sep = ';', na_values = ['?', ''])

# The `na_values` argument passed to the `read_csv` function:
#   1. Converts string entries containing '?' into NaN (missing values).
#   2. Enables numeric columns to be parsed as floats instead of objects.


#===========================================================================================
        #%% Data Overall View
class data_overall_view:
    def __init__(self, data):
        """_Print a summary of dataset_"""
        self.data = data

    def data_type_summary(self):
        """_Information of features type_"""
        return self.data.info()

    def statical_summary(self):
        """_A brief statical information of data_"""
        print(f'\nNumericalStatistic:\n{self.data.describe()}')
        print(( f'\nCategoricalStatistic:\n {self.data.describe(include = object)}'))

    def data_null_value(self):
        """_Features with the number of their Nan values_"""
        print(f'\nNumber of NaNs:\n{self.data.isnull().sum()}')

datacheck = data_overall_view(df)
datacheck.data_type_summary()
datacheck.statical_summary()
datacheck.data_null_value()
 
 
#===========================================================================================
        #%% NaN values visualization 
def missing_values(data, thresh = None, color = None, edgecolor = 'black'): 
    '''Plot percentage of NaN values for each feature.'''
    plt.figure(figsize = (20, 8)) 
    percentage = (data.isnull().mean()) * 100 
    percentage.sort_values(ascending = False).plot.bar(color = color, edgecolor = edgecolor) 

    plt.axhline(y = thresh, color = 'darkred', linestyle = '-') 
    plt.text(len(data.isnull().sum()/len(data))/1.7, thresh + 0.091, f'Columns with more than {thresh}% missing values', fontsize=15, color='darkred', ha='left', va='top') 
    plt.text(len(data.isnull().sum()/len(data))/1.7, thresh - 0.04, f'Columns with less than {thresh}% missing values', fontsize=15, color='darkgreen', ha='left', va='top') 
    
    plt.xlabel('Columns', size = 20, weight = 'bold') 
    plt.ylabel('Missing values percentage(%)', size = 20, weight = 'bold') 
    plt.yticks(size = 20) 
    plt.xticks(size = 20)
    plt.title('Missing values percentage per column', fontsize = 20, weight = 'bold') 

missing_values(df.iloc[:,2:], thresh = 2, color = color_pal) 




#%%
