
#################################### DATA ANALYSIS ####################################
#### NUMERICAL DATA CORRELATION
#### DISTRIBUTION OF FEATURES
#### CONSUMPTION BASED ON THE DATE
#### PATTERNS FOR CONSUMPTION OVER DIFFERENT TIME SCALES
#### CONSUMPTION DISTRIBUTION ACROSS DIFFERENT TIME SCALES 
#### MOVING AVERAGE OVER A YEAR
#### SEASONAL PATTERNS
#### LINEAR REGRESSION MODEL FOR TREND AND ENERGY AS FEATURE AND TARGET
#### CENTERING THE DATASET




#===========================================================================================
    #%% Numerical Data Correlation
def data_correlation(data_corr, threshold):
    """Plot correlation between numerical data."""
    corr_threshold = data_corr[(data_corr > threshold) | (data_corr < -threshold)] 
    plt.figure(figsize=(10, 8)) 
    sns.heatmap(corr_threshold, annot=True, cmap=color_pal, fmt=".3f", linewidths=0.5, cbar_kws={'shrink': .5},  annot_kws={'size': 8});
    plt.title('Correlations Among Features', weight='bold')
    plt.show()

corr_mat= df.drop(columns=['Date','Time']).corr() 
data_correlation(data_corr=corr_mat, threshold=0.3)



#===========================================================================================
    #%% Distibution of features
def data_distribution(data, columns):
    """Plot distribution of features"""
    fig, axe = plt.subplots(3, 3, figsize=(12, 10))
    color_cycle = cycle(color_pal)
    for (col, unit), ax in zip(columns.items(), axe.ravel()): 
            sns.histplot(data, x=col, ax=ax, color=next(color_cycle), bins=200)
            ax.set_xlabel(f'{col} ({unit})', weight='bold')
    fig.suptitle('Distribution of data', fontsize=20, weight='bold')        
    plt.tight_layout() 
    plt.show()


features = {'Global_active_power':'Wh',
        'Global_reactive_power':'Wh',
        'Voltage':'V', 
        'Global_intensity':'A', 
        'Sub_metering_1':'Wh',
        'Sub_metering_2':'Wh',
        'Sub_metering_3':'Wh',
        'Main_energy_wh':'Wh', 
        'power_factor':''}
data_distribution(data=df, columns=features)



#===========================================================================================
    #%% Consumption based on the date
def consumption_timeseries(data, columns, units):
    """Plot consumption features over time"""
    n = len(columns)
    fig, axe = plt.subplots(n, 1, figsize=(30, 28))
    color_cycle = cycle(color_pal)
    for col, unit, ax in zip(columns,units,axe.ravel()): 
            data.plot(y=col, color=next(color_cycle), ms=1, ax=ax, fontsize=30)
            ax.set_title(f'{col} over time ({unit})', fontsize=25, weight='bold')
    plt.tight_layout()
    ax.set_ylabel(unit, weight='bold')
    plt.show()

columns= ['Global_active_power', 'Global_reactive_power', 'Main_energy_wh', 'Global_intensity', 'Voltage' ]
units = ['Wh', 'Wh', 'Wh', 'A', 'V']
consumption_timeseries(data=df, columns=columns , units=units )



#===========================================================================================
    #%% Patterns for Consumption over different time scales
def consumption_pattern(data):
    """Plot consumption patterns at yearly, monthly and daily scales"""
        
        # Monthly Energy Consumption (wh)
    # '01-01-2010' -> '01-01-2011'
    data[(data.index > '01-01-2009') & (data.index < '01-01-2010')].plot(
        y='Main_energy_wh', style='.', figsize=(40,10), color=color_pal[0], fontsize=30)
    plt.title('Monthly Energy Consumption (wh)', fontsize=30, weight='bold')
    plt.show()

    # '08-01-2009' -> '09-01-2009' 
        # Daily Energy Consumption (wh)
    data[(data.index > '05-01-2009') & (data.index < '06-01-2009')].plot(
        y='Main_energy_wh', style='.', figsize=(30,10), color=color_pal[1], fontsize=30)
    plt.title('Daily Energy Consumption (wh)', fontsize=30, weight='bold')
    plt.show()

    # '04-28-2007' ->  '04-29-2007'
        # Hourly Energy Consumption (wh)
    data[(data.index > '04-01-2009') & (data.index < '04-02-2009')].plot(
        y='Main_energy_wh', style='.', figsize=(30,10), color=color_pal[2], fontsize=30)
    plt.title('Hourly Energy Consumption (wh)', fontsize=30, weight='bold')
    plt.show()

consumption_pattern(data=df)



#===========================================================================================
    #%% Consumption distribution across different time scales
# Create time series features
df_time_series = create_time_series_features(df)

def consumption_boxplot(data):
    """Plot boxplots of energy consumption across different time periods"""
    fig, axe = plt.subplots(nrows=4, ncols=1, figsize=(20, 15))
    axe = axe.flatten()
    for ax, period in zip(axe, ['year', 'month', 'hour']):
        sns.boxplot(data, x=period, y='Main_energy_wh', palette='Pastel1', ax=ax)
        ax.set_title(f"Energy consumption by {period}")

    sns.boxplot(data=df_time_series, x='dayofweek', y='Main_energy_wh', hue='season', ax=axe[3], linewidth=1, palette='Pastel1')
    axe[3].set_title("Energy consumption by Day of the week", weight='bold')
    axe[3].set_xlabel("Day of week")
    axe[3].set_ylabel("Energy (Wh)")
    axe[3].legend(bbox_to_anchor=(1, 1))    

    plt.tight_layout()
    plt.show()

consumption_boxplot(df_time_series)



#===========================================================================================
    #%% Moving average over a year
# Calculate the moving average over a year(how the data generally moves if we reduce the noise).
df_time_series = create_time_series_features(df)
def moving_average_function(energy_data, column):
    moving_average = energy_data[column].rolling(
        window=30*24*60,     # 30-day window 
        center=True,         # puts the average at the center of the window
        min_periods=(30*24*60)//2  # choose about half the window size
    ).mean()

    fig,ax = plt.subplots(figsize=(20, 9))
    energy_data.plot(y=column, style='.', ax=ax, ms=0.1, color=color_pal[1], fontsize = 20)
    moving_average.plot(linewidth=4, ax=ax, legend=False, color=color_pal[0], fontsize = 20)

    plt.title("Energy Consumption - 30-day moving average (Monthly trend)", fontsize=20, weight='bold')
    plt.show()

moving_average_function(df_time_series, 'Main_energy_wh')



#===========================================================================================
    #%% Plot seasonal patterns
def seasonal_plot(X, y, period, freq, ax = None): 
    if ax is None: 
        _ , ax = plt.subplots()
    palette = sns.color_palette('husl', n_colors = X[period].nunique())
    ax = sns.lineplot(x=freq, 
                     y=y, 
                     hue=period, 
                     data=X, 
                     errorbar=('ci', False),
                     ax=ax, 
                     palette=palette,
                     legend= False)
    ax.set_title(f'Seasonal Plot ({period}/{freq})', weight='bold')
    for line, name in zip(ax.lines, X[period].unique()): 
        y_ =  line.get_ydata()[-1]   #Gets the last y-value of the current line
        ax.annotate(name, 
                xy = (1, y_), 
                xytext = (6, 0),
                color = line.get_color(), 
                xycoords=ax.get_yaxis_transform(),
                textcoords="offset points",
                size=14,
                va="center")
    return ax                


df_time_series = create_time_series_features(df)
def seasonal_plots():
    fig, ax = plt.subplots(5, 1, figsize=(10, 30))
    seasonal_plot(df_time_series, y="Main_energy_wh", period="dayofweek", freq="hour", ax = ax[0])
    seasonal_plot(df_time_series, y="Main_energy_wh", period="week", freq="dayofweek", ax=ax[1])
    seasonal_plot(df_time_series, y="Main_energy_wh", period="month", freq="dayofmonth", ax=ax[2])
    seasonal_plot(df_time_series, y="Main_energy_wh", period="year", freq="quarter", ax=ax[3])
    seasonal_plot(df_time_series, y="Main_energy_wh", period="year", freq="month", ax=ax[4])

seasonal_plots()



#===========================================================================================
    #%% Create all the new features 
df = create_fourier_features(df)
df = create_time_series_features(df)
df = create_lag_features(df)


# NOTE: Here dataset should be processed to handel NaN values; Run data-processing before this modeling.
#===========================================================================================
    #%% Linear Regression model for trend and energy as feature and target 
def linear_reg_trend(data):
    # Create Feature and target for Linear Regression
    X = data.loc[:, ['trend']] #feature
    y = data.loc[:, 'Main_energy_wh'] #target
    # Create and train the model to find the trend in data wrt time.
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    # Make Prediction over dataset
    y_pred = pd.Series(lin_reg.predict(X), index=X.index)
    # Plot Regression Model 
    fig, ax = plt.subplots(figsize=(30, 10))
    df.plot(y='Main_energy_wh', style='.', ax=ax, color=color_pal[0], ms=1)
    y_pred.plot(ax=ax, color='black', lw=2, fontsize=25)
    ax.legend(['actual data', 'trend'])
    plt.title('Energy Consumption by Hour use in Wh and trend', fontsize = 25, weight = 'bold')
    plt.show()

    # Making future trend predictions and plot it
    future = pd.date_range('2010-11-26', '2011-11-26', freq='1h')
    future_df = pd.DataFrame(index=future)
    future_df['trend'] = np.arange(df['trend'].max() + 1, df['trend'].max() + len(future_df.index) + 1)
    print(future_df)
    # Plotting future predictions 
    y_future_trend_pred = pd.Series(lin_reg.predict(future_df), index=future_df.index)
    fig, ax = plt.subplots(figsize=(30, 10))
    df.plot(y='Main_energy_wh', style='.', ax=ax, color=color_pal[0], ms=1)
    y_pred.plot(ax=ax, color='black', lw=2, fontsize=25)
    y_future_trend_pred.plot(ax=ax, color=color_pal[3], lw=3, fontsize=25)
    ax.legend(['actual data', 'trend'])
    plt.title('Energy Consumption by Hour use in Wh', fontsize=25)
    plt.show()

linear_reg_trend(data=df)


    #%% Detrend the data
# df['detrended_EC_MW'] = df['Main_energy_wh'] - y_pred
# fig, ax = plt.subplots(figsize=(30, 10))
# df['detrended_EC_MW'].plot(style='.', ax=ax, color=color_pal[0], ms=1)
# plt.title(' Energy Consumption detrended')


#===========================================================================================
    #%% Centering the dataset
def centered_the_data(data):
    # Plot mean of the consumption based on itself
    fig, ax = plt.subplots(figsize=(20, 10))
    data.plot(y='Main_energy_wh', style='.', ax=ax, color=color_pal[0], ms=1)
    plt.axhline(y = data['Main_energy_wh'].mean(), color='black', linestyle='-', lw=3)
    ax.legend(['actual data', 'mean'])
    plt.title('Main energy consumption(wh) by Hour use and the mean value', weight='bold')

    # Centered the data
    data['centered_Main_energy_wh'] = data['Main_energy_wh'] - data['Main_energy_wh'].mean()
    fig, ax = plt.subplots(figsize=(20, 10))
    data['centered_Main_energy_wh'].plot(style='.', ax=ax, color=color_pal[0], ms=1)
    plt.title('Main energy consumption(Wh) centered around 0', weight='bold')
    plt.show()

centered_the_data(df)






