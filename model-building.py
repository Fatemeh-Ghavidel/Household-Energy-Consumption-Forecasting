#################################### MODEL BUILDING ####################################
#### NAN VALUES MITIGATION
#### FEATURE ENGINEERING AND DATA PROCESSING 
#### DATA PREPARATION FOR MODELING
#### LINEAR REGRESSION 
#### XGBOOST MODEL


#===========================================================================================
        #%% Nan Values mitigation
df['Main_energy_wh'] = df['Main_energy_wh'].interpolate(
    mehtod = 'time', 
    limit_direction= 'both')



#===========================================================================================
        #%% Feature eng and data processing 
df = create_fourier_features(df)
df = create_time_series_features(df)
df = create_lag_features(df)


df['lag1year'] = df['lag1year'].fillna(df['Main_energy_wh'].median())
df['lag2year'] = df['lag2year'].fillna(df['Main_energy_wh'].median())
df['lag3year'] = df ['lag3year'].fillna(df['Main_energy_wh'].median())



#===========================================================================================
        #%%Data Prepration for modeling 
#Categorical features conversion
df = pd.get_dummies(df)
#Features and target column names
target = 'Main_energy_wh'
features_linear_reg = ['trend', 'sin(1,freq=B)', 'cos(1,freq=B)', 'sin(2,freq=B)',
       'cos(2,freq=B)', 'sin(3,freq=B)', 'cos(3,freq=B)', 'sin(4,freq=B)',
       'cos(4,freq=B)', 'sin(5,freq=B)', 'cos(5,freq=B)', 'sin(6,freq=B)',
       'cos(6,freq=B)', 'hour', 'month', 'year', 'dayofweek',
       'dayofyear', 'weekofyear', 'lag1year',  'lag2year', 'lag3year', ]
features_xgb_reg = ['trend', 'sin(1,freq=B)', 'cos(1,freq=B)', 'sin(2,freq=B)',
       'cos(2,freq=B)', 'sin(3,freq=B)', 'cos(3,freq=B)', 'sin(4,freq=B)',
       'cos(4,freq=B)', 'sin(5,freq=B)', 'cos(5,freq=B)', 'sin(6,freq=B)',
       'cos(6,freq=B)',  'hour', 'month', 'year', 'dayofweek',
       'dayofyear', 'weekofyear', 'lag1year',    'lag2year', 'lag3year', ]

#Train/Test split 
train = df.loc[df.index < '26-11-2009'].copy()
test = df.loc[df.index >= '26-11-2009'].copy()

#Plot train/test split 
def plot_train_test_split(train, test):
        fig, ax = plt.subplots(figsize=(30,10))
        train.plot(y = 'Main_energy_wh', style='.', ax=ax, label='Training Set', color=color_pal[0], fontsize=30)
        test.plot(y = 'Main_energy_wh', style='.', ax=ax, label='Test Set', color=color_pal[1], fontsize=30)
        ax.legend(["Training Set", "Testing Set"])
        ax.axvline(test.index.min(), color='black', ls='--')
        plt.title("Train / Test Split", fontsize=25, weight='bold')
        plt.show()

plot_train_test_split(train, test)

#Plot time series split
def time_series_split(train):
        tss = TimeSeriesSplit(n_splits=5, test_size=5*60*365*1, gap=5*60)
        train = train.sort_index()

        fig, axs = plt.subplots(5, 1, figsize=(15,15), sharex=True)

        fold=0
        for train_idx, eval_idx in tss.split(train):
                train_set=train.iloc[train_idx]
                eval_set=train.iloc[eval_idx]
                train_set['Main_energy_wh'].plot(ax=axs[fold], label="Training Set",color = color_pal[0],)
                eval_set['Main_energy_wh'].plot(ax=axs[fold], label='Test Set', color = color_pal[1],)
                axs[fold].axvline(eval_set.index.min(), color='black', ls='--')
                fold += 1

time_series_split(train)




#===========================================================================================
        #%% Linear Regression model
# Linear regression  training 
def train_linear_regression(train, features_linear_reg, target):
        
        X_train = train[features_linear_reg]
        y_train = train[target] 
        X_test = test[features_linear_reg]
        y_test = test[target]

        #Train the model 
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)

        #Make prediction with the model on training set
        y_train_pred = pd.Series(lin_reg.predict(X_train), index=X_train.index)
        #Make prediction with the model on test set
        y_test_pred = pd.Series(lin_reg.predict(X_test), index=X_test.index)
        
        #Plot Test Prediction 
        ax = test[['Main_energy_wh']].plot(figsize=(30, 10), color = color_pal[0])
        y_test_pred.plot(ax=ax, color = color_pal[1], fontsize=25)
        plt.legend(['Actual Values', 'Predicted Data'])
        plt.title("Actual Values vs Test Predictions", weight='bold', fontsize=25)
        plt.show()

        return y_test, y_test_pred

y_test, y_test_pred  =train_linear_regression(train, features_linear_reg, target)

# Linear regression evaluation
def linear_reg_eval(y_test, y_test_pred):
        # MAE: Average error in Wh 
        mae = mean_absolute_error(y_test, y_test_pred)
        # RMSE: Penalizes large errors more heavily (in Wh)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        print(f"MAE: {mae:.2f} Wh")
        print(f"RMSE: {rmse:.2f} Wh")

linear_reg_eval(y_test, y_test_pred)




#===========================================================================================
        #%% XGboost Model  
#Initial model training
def init_xgboost_training(train):
        tss = TimeSeriesSplit(n_splits=5, test_size=5*24*365*1, gap=5*24 )
        train = train.sort_index()

        scores= [] 
        preds = []

        for train_idx, eval_idx in tss.split(train):
                train_set = train.iloc[train_idx]
                eval_set = train.iloc[eval_idx]

                X_train = train_set[features_xgb_reg]
                y_train = train_set[target]
                X_eval = eval_set[features_xgb_reg]
                y_eval = eval_set[target]

                #define the model
                xgb_reg_1 = xgb.XGBRegressor(n_estimators=1000,
                                        max_depth=4, booster='gbtree',
                                        early_stopping_rounds=50, base_score=0.5,
                                        objective='reg:squarederror', learning_rate=0.01)

                #train the model
                xgb_reg_1.fit(X_train, y_train,
                        eval_set = [(X_train, y_train), (X_eval, y_eval)],
                        verbose=100)
        
                y_pred = xgb_reg_1.predict(X_eval)
                preds.append(y_pred)
                score = mean_absolute_error(y_eval, y_pred)
                scores.append(score)
        

        print(f'Average of scores for Cross Validation: {np.mean(scores):0.4f}')
        print(f'Fold Scores: {scores}')
        print(f'Average MAE: {np.mean(scores):.4f}')
        print(f'Fold MAEs: {scores}')

init_xgboost_training(train)


#Final model training with all training data
def final_xgboost_training(train): 

        X_train_all = train[features_xgb_reg]
        y_train_all = train[target]

        xgb_reg = xgb.XGBRegressor(base_score=0.5, 
                        booster='gbtree', 
                        n_estimators=400, 
                        objective='reg:squarederror', 
                        max_depth=3, 
                        learning_rate=0.01, 
                        early_stopping_rounds=50 )

        xgb_reg.fit(X_train_all,
                y_train_all, 
                eval_set = [(X_train_all, y_train_all)],
                verbose = 100)
        return xgb_reg 

xgb_reg_2 = final_xgboost_training(train)

#Learned feature importance 
def plot_feature_importance(xgb_reg):
        feat_imp = pd.DataFrame(data=xgb_reg.feature_importances_,
                                index = xgb_reg.feature_names_in_,
                                columns=['importance'])
        feat_imp.sort_values('importance').plot(kind='barh', color=color_pal[0])
        plt.title("Feature Importances")
        plt.show()

plot_feature_importance(xgb_reg_2)


#Make prediction with final XGBoost model on test set
def pred_xgboost_model(test, features_xgb_reg, xgb_reg, target):

        X_test = test[features_xgb_reg]
        y_test = test[target]

        test['predictions'] = xgb_reg.predict(X_test)

        #Plot prediction of test set
        ax = y_test.plot(figsize=(30,10), color=color_pal[0])
        test['predictions'].plot(ax=ax,  color=color_pal[1], fontsize=25)
        plt.legend(['Actual Values', 'Predicted Data'])
        plt.title("Actual Values vs Predictions", weight='bold')
        plt.show()
        #Plot montly prediction of test set
        ax = y_test.loc[(y_test.index > '01-01-2010') & (y_test.index < '02-01-2010')].plot(figsize=(15, 5),  color =  color_pal[0])
        test.loc[(test.index > '01-01-2010') & (test.index < '02 -01-2010')]['predictions'].plot(style='.',  color =  color_pal[1])
        plt.legend(['Actual Data', 'Predicted Values'])
        plt.title("Monthly Actual Values vs Predictions", weight='bold')
        plt.show()

        return y_test

y_test = pred_xgboost_model(test, features_xgb_reg, xgb_reg_2, target='Main_energy_wh')


#Evaluation of the model 
def xgboost_eval(test, y_test, target):

        mae = mean_absolute_error(y_test, test['predictions'])
        rmse = np.sqrt(mean_squared_error(y_test, test['predictions']))

        print(f'MAE Score on test set: {mae:0.2f}')
        print(f'RMSE Score on test set: {rmse:0.2f}')

        test['error'] = np.abs(test[target] - test['predictions'])
        test['date'] = test.index.date
        print('Worst Predictions:')
        display(test.groupby(['date'])['error'].mean().sort_values(ascending=False).head())
        print('\nBest Predictions:')
        display(test.groupby(['date'])['error'].mean().sort_values(ascending=True).head())

xgboost_eval(test, y_test, target='Main_energy_wh')





# %%
