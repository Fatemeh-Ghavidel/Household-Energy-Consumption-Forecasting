
#################################### FEATURE ENGINEERING ####################################
#### UNIT CHANGE
#### CREATE CONSEPTUAL FEATURES 
#### CREATE DATE-TIME FEATURE AS INDEX
#### CREATE TEMPORAL FEATURES: TimeSeries Features, Fourier Features and Lag Features,



#========================================================================
    #%%Unit Change(kW-> wh)
df['Global_active_power'] = df['Global_active_power'] *1000/60 
df['Global_reactive_power'] = df['Global_reactive_power'] *1000/60
  


#========================================================================
    #%%Create Conseptual Features
#Submetring to the active power 
total_sub3 = df['Sub_metering_3'].sum()
total_sub2 = df['Sub_metering_2'].sum()
total_sub1 = df['Sub_metering_1'].sum()
total_global = df['Global_active_power'].sum()

print(f"Sub_metering_3: {total_sub3 / total_global * 100:.2f}%")
print(f"Sub_metering_2: {total_sub2 / total_global * 100:.2f}%")
print(f"Sub_metering_1: {total_sub1 / total_global * 100:.2f}%")


#Create new consumtion feature without considering submetrings (wh)
df['Main_energy_wh'] = df['Global_active_power'] - df['Sub_metering_1']- df['Sub_metering_2']-df['Sub_metering_3']
#Remove negative values of this feature
df['Main_energy_wh'] = df['Main_energy_wh'].clip(lower = 0)

#Create power factor
Apparent_power = df['Voltage'] * df['Global_intensity']
df['power_factor'] = (df['Global_active_power'] ) / Apparent_power




#========================================================================
    #%%Create Data-time index
#Combine two features of `Date` and `Time`
df['Date_time'] = df['Date'] + ' ' + df['Time']

#Convert date_time as (obj) to (pd.datetime) then set that to index
df['Date_time'] = pd.to_datetime(df['Date_time'], format = '%d/%m/%Y %H:%M:%S')    #16/12/2006 17:25:00
df = df.set_index('Date_time')



#========================================================================
    #%%Create temporal features 

#Create time series features
def create_time_series_features(data): 
    """ Creates time series features based on datetime index"""
    data = data.copy()

    data['date'] = data.index
    data['hour'] = data.index.hour
    data['month'] = data.index.month
    data['year'] = data.index.year
    data['dayofweek'] = data.index.dayofweek
    data['dayofyear'] = data.index.dayofyear
    data['quarter'] = data.index.quarter
    data['dayofmonth'] = data['date'].dt.day
    data['weekofyear'] = data['date'].dt.isocalendar().week
    data['week'] = data["weekofyear"].astype(str)

    data['date_offset'] = (data.date.dt.month*100 + data.date.dt.day - 320)%1300
    data['season'] = pd.cut(data['date_offset'], [-1, 300, 602, 900, 1301],
                         labels=['Spring', 'Summer', 'Fall', 'Winter']) 

    return data


#Create fourier features 
def create_fourier_features(data): 
    fourier = CalendarFourier(freq ='B', order = 6)
    data = data.copy()
    dp = DeterministicProcess(
        index=data.index,
        constant=True,   # dummy feature for bias (y-intercept)
        order=1,       # trend (order 1 means linear)
        additional_terms=[fourier],     # annual seasonality (fourier)
        drop=True,    # drop terms to avoid collinearity
    )
    X = dp.in_sample() # create features for dates in df.index
    X['Main_energy_wh'] = data['Main_energy_wh']
    return X


#Create lag features
def create_lag_features(data): 
    data = data.copy()
    target_map = data['Main_energy_wh'].to_dict()
    data['lag1year'] = (data.index - pd.Timedelta('364 days')).map(target_map)
    data['lag2year'] = (data.index - pd.Timedelta('728 days')).map(target_map)
    data['lag3year'] = (data.index - pd.Timedelta('1092 days')).map(target_map)
    return data


#%%
