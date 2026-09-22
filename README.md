# **Project Title:** 

Forecasting Household Energy Consumption Using Temporal Feature Engineering and Gradient Boosting

# Problem Definition 

Household electricity consumption is highly irregular and nonlinear, driven by a mix of daily routines, weekly patterns, seasonal weather effects, and unpredictable appliance usage. This variability makes it difficult for utilities and consumers to anticipate demand, plan capacity, or manage costs.

The goal of this project is to forecast household energy consumption (Global Active Power, measured in `Wh`) using historical minute-level measurements from 2006 to 2010. The core challenge is to capture the underlying temporal structure, daily, weekly, and yearly cycles, while preventing data leakage and overfitting in a dataset of over 2 million observations.

The problem is framed as a supervised regression task:
- **Input (features):** Time-based features (hour, day, week, month, season), Fourier terms for smooth seasonality, and lagged values of the target variable.
- **Output (target):** Energy consumption at a given minute.
- **Evaluation:** Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) on unseen future data, using time-series cross-validation.

# **Dataset:** 

## Dataset Overview

This study uses a household energy consumption dataset that recorded measurements at one-minute intervals between 2006 and 2010. The dataset is publicly available at the following link: [individual+household+electric+power+consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption "Link")

It contains the following features:
- Global Active Power (`kW`)
- Global Reactive Power (`kVAR`)
- Voltage (`V`)
- Global Intensity (`A`)
- Sub-metering 1 (`Wh`)
- Sub-metering 2 (`Wh`)
- Sub-metering 3 (`Wh`)

After loading the dataset, an initial overview is performed. This overview summarizes the data types, memory usage, descriptive statistics for both numerical and categorical features, and the number of missing values in each column.

The missing-value counts indicate that all energy-related features share the same number of NaN values, suggesting the gaps occur at identical timestamps across columns. To handle these missing values, their exact locations and durations will be examined in subsequent steps.

## Dataset Explanation 

**Global Active Power (`kW`)**
The total real power consumed by the household; the power that actually performs useful work (lighting, heating, running motors) and is billed by the utility. It is a rate of energy use, measured in kilowatts (`kW`). To convert it into energy (`Wh` or `kWh`), it is multiplied by the time interval:
`Energy (Wh)=Global Active Power (kW)×1000/60​`
for a 1-minute sampling interval.

**Global Reactive Power (`kVAR`)**
The portion of power that does not perform useful work but is required to sustain the magnetic fields in motors, transformers, and inductive loads. It is measured in kilovolt-amperes reactive (`kVAR`) and is not billed directly, but it affects the overall efficiency and quality of the electrical supply.

**Sub-metering 1, 2, 3 (`Wh`)**
Energy consumption (in watt-hours) recorded by three dedicated sub-meters, each monitoring a specific circuit group:
* Sub-metering-1: Kitchen (dishwasher, oven, microwave)
* Sub-metering-2: Laundry room (washing machine, dryer, refrigerator, light)
* Sub-metering-3: Electric water heater and air conditioner           
These sub-meters cover only a fraction of total household consumption; the remainder is captured by the difference between Global Active Power and the sum of the three sub-meters.

**Voltage (`V`)**
The electrical potential difference that drives current through the circuit; the push that moves electrons. In residential settings it is typically 120 `V` (North America) or 230–240 `V` (Europe). Voltage in this dataset is measured in volts (`V`) and usually fluctuates within a narrow band around its nominal value.

**Global Intensity / Current (`A`)**
The rate of electric charge flow through the main circuit, measured in amperes (`A`). It reflects how much electricity is being drawn at any instant; more appliances running means higher current. It is related to power and voltage by:
`P=V×I` 
where `P` is power (watts), `V` is voltage (volts), and `I` is current (amperes).

# File Execution Order

1. `imports`
2. `data_ingestion`
3. `feature-engineering`
4. `data-analysis`
5. `data_processing`
6. `model_building`


       






`The Linear Regression model serves as a **benchmark**. Its performance reflects how much of the consumption signal can be explained by a purely linear combination of temporal features. If XGBoost (a nonlinear model) significantly outperforms it, that indicates the presence of nonlinear interactions between features — for example, the effect of hour depends on season, or the effect of dayofweek depends on month. This comparison justifies the use of tree-based models in the later stages of the project.`
