import os.path
import urllib.request
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from math import pi, cos

url_mt = "https://stacks.stanford.edu/file/druid:py883nd2578/MT-clean.csv.gz"
url_vt = "https://stacks.stanford.edu/file/druid:py883nd2578/VT-clean.csv.gz"

file_name_mt = "MT-clean.csv.gz"
file_name_vt = "VT-clean.csv.gz"

def download_data(file_name, url):
    if not os.path.isfile(file_name):
        urllib.request.urlretrieve(url, file_name)
        
download_data(file_name_mt, url_mt)
download_data(file_name_vt, url_vt)
  
cols_mt = ['stop_date', 'stop_time', 'driver_gender', 'violation', \
        'out_of_state', 'vehicle_year', 'lat', 'lon', 'is_arrested',
        'county_fips', 'county_name']

cols_vt = ['stop_time', 'violation']

data_mt = pd.read_csv(file_name_mt, usecols=cols_mt, low_memory=False)
data_vt = pd.read_csv(file_name_vt, usecols=cols_vt, low_memory=False)

mask_male = data_mt['driver_gender'] == 'M'

prop_male = mask_male.sum()/data_mt.shape[0]

# Comment: we are not given any specific instruction about how 
# to deal with 'nan' values. For this reason, I decided to ignore 
# them.

print("\nThe proportion of male drivers is {}.".format(prop_male))

# The in- and out-of-state masks:

mask_in = data_mt['out_of_state'] == False
mask_out = data_mt['out_of_state'] == True

# The total numbers of in- and out-of-state drivers

n_in = mask_in.sum()
n_out = mask_out.sum()


# The number of in- and out-of-state drivers arrested

n_in_ar = data_mt.is_arrested[mask_in].sum()
n_out_ar = data_mt.is_arrested[mask_out].sum()


# The proportion of in- and out-of-state drivers arrested

prop_in_ar = n_in_ar / n_in
prop_out_ar = n_out_ar / n_out

# The ratio

r_in_out = prop_out_ar/prop_in_ar

print("\nOut-of-state drivers are {} times more likely to be arrested.".\
      format(r_in_out))

is_arrested_male = data_mt.is_arrested[mask_male]
is_arrested_oos = data_mt.is_arrested[mask_out]

male_tab = pd.crosstab(index=is_arrested_male, columns="count")
oos_tab = pd.crosstab(index=is_arrested_oos, columns="count")

observed = oos_tab

male_ratios = male_tab/len(is_arrested_male)  # Get population ratios

expected = male_ratios * len(is_arrested_oos)   # Get expected counts

 # Find the p-value

p_value = stats.chisquare(f_obs= observed, f_exp= expected)[0][0]

print("\nThe value of the chi-square test statistic is {}.".format(p_value))

# If the order of observed and expected is switched

observed = male_tab

oos_ratios = oos_tab/len(is_arrested_oos)

expected = oos_ratios * len(is_arrested_male)

p_value = stats.chisquare(f_obs= observed, f_exp= expected)[0][0]

print("\nThe alternative (opposite order) value of the ")
print("chi-square test statistic is {}.".format(p_value))

# Mask for speeding violations

mask_speed = data_mt.violation.str.contains("Speeding").fillna(False)

# The proportion of speeding violations

prop_speed = mask_speed.sum()/data_mt.shape[0]

print("\nThe proportion of speeding violations is {}.".format(prop_speed))

# Masks for DUI violations in Montana and Vermont

mask_dui_mt = data_mt.violation.str.contains("DUI").fillna(False)
mask_dui_vt = data_vt.violation.str.contains("DUI").fillna(False)

# The proportion of DUI violations

prop_dui_mt = mask_dui_mt.sum()/data_mt.shape[0]
prop_dui_vt = mask_dui_vt.sum()/data_vt.shape[0]

r_dui = prop_dui_mt / prop_dui_vt

print("\nIt is {} times more likely that".format(r_dui))
print("a trafic stop in Montana will result")
print("in a DUI violation than a trafic stop in Vermont.")

# Parsing dates

data_mt['stop_data'] = pd.to_datetime(data_mt.stop_date, format='%Y-%m-%d')

# The stop year 

data_mt['year'] = data_mt['stop_data'].dt.year

# Replace non-numeric values with 'nan' in 'vehicle_year'

data_mt['vehicle_year'] = data_mt['vehicle_year'].\
                            replace(['NON-', 'UNK'], np.nan)
                            
data_mt['vehicle_year'] = pd.to_datetime(data_mt.vehicle_year, format='%Y')

data_mt['vehicle_year'] = data_mt['vehicle_year'].dt.year

gp = data_mt.groupby(['year'])[['vehicle_year']].agg('mean').\
        reset_index().rename(columns={'vehicle_year': 'average_vehicle_year'})
        
X = gp.year.values.reshape(-1, 1)
y = gp.average_vehicle_year.values

reg = LinearRegression()
reg.fit(X, y)

y_pred = reg.predict(X)

plt.scatter(X, y,  color='black')
plt.plot(X, y_pred, color='blue', linewidth=3)
plt.show()

X_2020 = np.array([2020]).reshape(-1, 1)
y_2020 = reg.predict(X_2020)[0]

print("\nThe extrapolated, average manufacture year of vehicles")
print("involved in traffic stops in Montana in 2020 is {}.".format(y_2020))

p_value_year = stats.linregress(X.reshape(1, -1)[0], y)[3]

print("\nThe p-value of the linear regression is {}.".format(p_value_year))

# Extracting the hour information from 'stop_time'

data_mt[['hour', 'min']] = data_mt.stop_time.str.split(':', expand=True)
data_vt[['hour', 'min']] = data_vt.stop_time.str.split(':', expand=True)

data_mt['hour'] = pd.to_datetime(data_mt.hour, format='%H')
data_vt['hour'] = pd.to_datetime(data_vt.hour, format='%H')

data_mt['hour'] = data_mt['hour'].dt.hour
data_vt['hour'] = data_vt['hour'].dt.hour

data_mt.drop(['stop_time', 'min'], axis=1, inplace=True)
data_vt.drop(['stop_time', 'min'], axis=1, inplace=True)

stop_hours = pd.concat((data_mt.hour, data_vt.hour), axis=0)

hour_tab = stop_hours.value_counts().sort_values()

# The most and least frequent hours

mf_stops = hour_tab.iloc[-1]
lf_stops = hour_tab.iloc[0]

diff_stops = mf_stops - lf_stops

print("\nThe difference in the total number of stops")
print("that occurred in the most and least frequent")
print("hours is {}.".format(diff_stops))

# Selecting only rows needed to comput the area

data = data_mt[['county_name', 'lon', 'lat']].dropna()

# Filtering out unrealistic values of 'lon' and 'lat'
# (used Google maps to estimate the min and max values)

lat_min = 44.3
lat_max = 49.1

lon_min = -116.5
lon_max = -103.8

mask_lat = (data.lat > lat_min)&(data.lat < lat_max)
mask_lon = (data.lon > lon_min)&(data.lon < lat_max)

mask_coords = mask_lat & mask_lon

data = data[mask_coords]

data = data.groupby(['county_name'])[['lon', 'lat']].agg(['std', 'mean'])
                 
data.drop([('lon', 'mean')], axis=1, inplace=True)

# Convert to radians

data = data * pi / 180

# Earth's mean radius (in km)

r_earth = 6371
    
data['area'] = pi * data[('lat', 'std')] * r_earth * \
            data[('lon', 'std')] * r_earth * data[('lat', 'mean')].apply(cos)

max_area = data.area.sort_values().iloc[-1]

print("\nThe largest area is {}.".format(max_area))