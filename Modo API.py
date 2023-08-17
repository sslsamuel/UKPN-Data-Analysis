# Databricks notebook source
import requests
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

def generate_url_frequency(timestamp1, timestamp2):
    new_url = f"https://api.modo.energy/public/v1/frequency?date_from={timestamp1}&date_to={timestamp2}"
    return new_url

def generate_url_prices(timestamp1,timestamp2):
    new_url = f"https://api.modo.energy/public/v1/uk_prices?date_from={timestamp1}&date_to={timestamp2}"
    return new_url   

# COMMAND ----------

def scrape_modo_prices_data(headers,t_start,t_end):    #PRICES
    headers = {
    'X-Token': headers
    }

    out = []
    t_start= datetime.strptime(t_start, "%Y-%m-%d")
    ti = t_start
    t_end = datetime.strptime(t_end, "%Y-%m-%d")
    while ti < t_end:
        #build times
    
        date_str_1 = ti#f(ti), datetime.strftime('format')
        date_str_2 = ti + relativedelta(months = 1) #f(ti+1000)
        new_url = generate_url_prices(date_str_1, date_str_2) #year-month-day
        print(date_str_1, date_str_2)

        response = requests.get(new_url, headers=headers)

        response_json = response.json() 
        results = response_json['results']

        df = pd.DataFrame.from_dict(results)
        out.append(df)

        ti += relativedelta(months = 1)  #length timedelta object
        
    
    

    #Merge dataframes (concat)
    combined_dataframe = pd.concat(out, ignore_index=True)
    combined_dataframe = combined_dataframe.drop_duplicates()
    
    return combined_dataframe 
  



# COMMAND ----------

def scrape_modo_frequency_data(headers,t_start,t_end):    #FREQUENCY
    headers = {
    'X-Token': headers
    }

    out = []
    t_start= datetime.strptime(t_start, "%Y-%m-%d")
    ti = t_start
    t_end = datetime.strptime(t_end, "%Y-%m-%d")
    while ti < t_end:
        #build times
    
        date_str_1 = ti#f(ti), datetime.strftime('format')
        date_str_2 = ti + relativedelta(days = 1) #f(ti+1000)
        date_str_1 = date_str_1.strftime("%Y-%m-%d")
        date_str_2 = date_str_2.strftime("%Y-%m-%d")
        new_url = generate_url_frequency(date_str_1, date_str_2) #year-month-day
       
        response = requests.get(new_url, headers=headers)

        response_json = response.json() 
        results = response_json['results']

        df = pd.DataFrame.from_dict(results)
        out.append(df)

        ti += relativedelta(days = 1)  #length timedelta object
      
    
    
  
    #clean the excess that is > tend

    #Merge dataframes (concat/merge)
    combined_dataframe = pd.concat(out, ignore_index=True)
    combined_dataframe = combined_dataframe.drop_duplicates()
    return combined_dataframe 
  

# COMMAND ----------

def plot_prices(df):    #PRICES
    
    df['start_time'] = pd.to_datetime(df['start_time'])
    plt.figure(figsize=(15, 6))
    plt.plot(df['start_time'], df['price'])
    plt.xlabel('Month')
    plt.ylabel('Price, £/MWh')
    plt.title('Prices between 2020 - 2023')
    plt.grid(True) 
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.show()
plot_prices(scrape_modo_prices_data('API KEY','2022-07-01','2023-06-30'))

# COMMAND ----------

def merge_data_prices(df_storage,df_prices):            #PRICES
    df_storage = pd.read_csv(df_storage) 
    df_storage = df_storage[['storage1','timestamp']].copy()
    df_storage['timestamp'] = pd.to_datetime(df_storage['timestamp'], utc=True)

    df_prices['timestamp'] = df_prices['start_time']
    df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'], utc=True)

    # resample df_prices to 30 min resolution 
    df_prices = df_prices.set_index('timestamp')
    df_prices = df_prices.resample('30T').ffill()



    # Merge the two DataFrames based on the common key 'timestamp'
    merged_df = pd.merge(df_prices, df_storage, on='timestamp')
    return merged_df


# COMMAND ----------

def merge_data_frequency(df_storage,df_frequency):            #FREQUENCY
    df_storage = pd.read_csv(df_storage) 
    df_storage = df_storage[['storage1','timestamp']].copy()
    df_storage['timestamp'] = pd.to_datetime(df_storage['timestamp'], utc=True)

    #df_frequency['timestamp'] = df_frequency['start_time']
    df_frequency['timestamp'] = pd.to_datetime(df_frequency['timestamp'], utc=True)

    # resample df_prices to 30 min resolution 
    df_frequency = df_frequency.set_index('timestamp')
    df_frequency = df_frequency.resample('15S').ffill()



    # Merge the two DataFrames based on the common key 'timestamp'
    merged_df = pd.merge(df_frequency, df_storage, on='timestamp')
    return merged_df

# COMMAND ----------

def scatter_graph_prices(merged_df):    #PRICES
 
    # Plot the scatter graph
    
    plt.scatter(merged_df['price'], merged_df['storage1'], s=1)
    plt.title('Price vs. Storage (07/2022 - 06/2023)')
    plt.xlabel('Price, £/MWh')
    plt.ylabel('Storage, MWh')
    plt.grid(True)  # Optional: Add grid lines to the plot
    plt.axvline(200, c='red')
    plt.axhline(0, c='red')
    plt.show()
    
scatter_graph_prices(merge_data_prices("/dbfs/FileStore/pidata_anon.csv",scrape_modo_prices_data('API KEY','2022-07-01','2023-06-30')))

# COMMAND ----------

def scatter_graph_frequency(merged_df):    #FREQUENCY
 
    # Plot the scatter graph
    print(merged_df)
    plt.scatter(merged_df['value'], merged_df['storage1'], s=1)
    plt.title('Frequency vs. Storage (07/2022 - 06/2023)')
    plt.xlabel('Frequency. Hz')
    plt.ylabel('Storage, MWh')
    plt.grid(True)  # Optional: Add grid lines to the plot
    plt.axvline(50, c='green')
    plt.axhline(0, c='green')
    plt.show()
    
scatter_graph_frequency(merge_data_frequency("/dbfs/FileStore/pidata_anon.csv",scrape_modo_frequency_data('API KEY','2023-01-01','2023-06-30')))

# COMMAND ----------

def scatter_hex_prices(merged_df):    
    g = sns.set(style="whitegrid")  # Set the style if desired
    g = sns.jointplot(x='price', y='storage1', data=merged_df, kind='hex',gridsize=20, bins=1000)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Price vs. Storage (07/2022 - 06/2023)')
    # Plot the scatter graph

    #plt.scatter(merged_df['price'], merged_df['storage1'], s=1)
    plt.xlabel('Price')
    plt.ylabel('Storage')
    plt.grid(True)  # Optional: Add grid lines to the plot
    plt.axvline(200, c='red')
    plt.axhline(0, c='red')
    plt.show()
scatter_hex_prices(merge_data_prices("/dbfs/FileStore/pidata_anon.csv",scrape_modo_prices_data('API KEY','2022-07-01','2023-06-30')))

# COMMAND ----------

def scatter_hex_frequency(merged_df):    
    g = sns.set(style="whitegrid")  # Set the style if desired
    g = sns.jointplot(x='value', y='storage1', data=merged_df, kind='hex',gridsize=20, bins=100)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Frequency vs. Storage (07/2022 - 06/2023)')
    # Plot the scatter graph

    #plt.scatter(merged_df['price'], merged_df['storage1'], s=1)
    plt.xlabel('Frequency, Hz')
    plt.ylabel('Storage, MWh')
    plt.grid(True)  # Optional: Add grid lines to the plot
    plt.axvline(50, c='green')
    plt.axhline(0, c='green')
    plt.show()
scatter_hex_frequency(merge_data_frequency("/dbfs/FileStore/pidata_anon.csv",scrape_modo_frequency_data('API KEY','2023-01-01','2023-06-30')))

# COMMAND ----------

#API Example

# Import python libraries
import requests
import pandas as pd

# Build the API URL
api_url = f"https://api.modo.energy/public/v1/uk_prices?date_from=2023-08-09&date_to=2023-08-09"
# Options are uk_prices, grid_frequency etc ------>

# Access with your credentials
apikey = 'API KEY'
headers = {'X-Token': apikey}

# Hit the API
response = requests.get(api_url, headers=headers)
# Get response into Pandas
df = pd.DataFrame.from_dict(response.json()['results'])

# Convert index to times
df['start_time'] = pd.to_datetime(df['start_time'])

# Play with the data
df.price.plot()
plt.xlabel('Time of day')
plt.ylabel('Price, £/MWh')
plt.title('Energy Prices  - 9 August 2023') 
plt.show()

#plt.plot(df['start_time'].dt.hour,df['price'])

# COMMAND ----------

columns_to_plot = ['EFR','Monthly FFR','Weekly FFR','Wholesale','BM','DCL','DCH','DML','DMH','DRL','DRH']

all_sites = ['broadditch','brook_farm','burwell','contego','cowstead','dollymans','mannington','wickham_market']


def get_battery_data(battery_site): 
    
    df= pd.read_csv(f"/dbfs/FileStore/shared_uploads/samuel.gagatek@ukpowernetworks.co.uk/{battery_site}_battery.csv")
    return df

# COMMAND ----------

def plot_battery_site(df, site, columns_to_plot):

    df = df.groupby("Settlement Period").mean()

    # get columns where sum doesn't = 0 
    col_sums = df.filter(columns_to_plot).sum(axis=0)
    filter_columns_to_plot = col_sums[col_sums != 0].index.tolist()
    filter_columns_to_plot.append('Settlement Period')
    
    settlement_period_avg = df.filter(filter_columns_to_plot).groupby("Settlement Period").mean()

    settlement_period_avg.plot(kind="line",figsize=(8, 5))

   

    #defining axes titles,attributes, and legend    
    plt.xlabel("Settlement Period")
    plt.ylabel("Revenue, £/MW")
    plt.title(f"Average income revenue throughout the day in July, 2023 for {site}")
    plt.legend(loc="center", bbox_to_anchor=(0.5, -0.4), frameon=True, ncol=3)
    plt.show(block = True)




# COMMAND ----------

def process_sites(site_names):
    for site in site_names:
        df = get_battery_data(site)
        plot_battery_site(df, site.title(),columns_to_plot)
process_sites(['brook_farm'])      

# COMMAND ----------

def process_aggregate_sites(site_names): 
    all_dfs = []
    for site in site_names: 
        df = get_battery_data(site)
        all_dfs.append(df)
    df_aggregated = pd.concat(all_dfs)
    plot_battery_site(df_aggregated,'All UKPN batteries' ,columns_to_plot)
    return df_aggregated
    


# COMMAND ----------

def plot_bar(df_aggregated): 
    df_mean = df_aggregated[columns_to_plot].mean()


    df_mean = df_mean.to_frame('Average Income')


    df_mean.sort_values(by = 'Average Income').plot(kind = 'bar')
    plt.xlabel('Source of Income')
    plt.ylabel('Revenue, £/MW')
    plt.title('Average UKPN battery sites revenue')
plot_bar(process_aggregate_sites(all_sites))

# COMMAND ----------

#df1 = pd.read_csv("/dbfs/FileStore/shared_uploads/samuel.gagatek@ukpowernetworks.co.uk/mannington_battery.csv")
#df2 = pd.read_csv("/dbfs/FileStore/shared_uploads/samuel.gagatek@ukpowernetworks.co.uk/wickham_market_battery.csv")
#df3 = pd.read_csv("/dbfs/FileStore/shared_uploads/samuel.gagatek@ukpowernetworks.co.uk/cowstead_battery.csv")
#df4 = pd.read_csv("/dbfs/FileStore/shared_uploads/samuel.gagatek@ukpowernetworks.co.uk/dollymans_battery.csv")
#df5 = pd.read_csv("/dbfs/FileStore/shared_uploads/samuel.gagatek@ukpowernetworks.co.uk/brook_farm_battery.csv")
#df6 = pd.read_csv("/dbfs/FileStore/shared_uploads/samuel.gagatek@ukpowernetworks.co.uk/burwell_battery.csv")
#df7 = pd.read_csv("/dbfs/FileStore/shared_uploads/samuel.gagatek@ukpowernetworks.co.uk/contego_battery.csv")
#df8 = pd.read_csv("/dbfs/FileStore/shared_uploads/samuel.gagatek@ukpowernetworks.co.uk/broadditch_battery.csv")

# COMMAND ----------

process_aggregate_sites(all_sites)
