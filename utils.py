import pandas as pd
import pickle
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def preprocess_data(values):
    # values = [[year ,month ,store_nbr ,family,city , state , type_ ,cluster]]
    data_df = pd.DataFrame(values,columns = ['Labour', 'Transactions', 'Capital','Store Cluster', 'Date', 'Holiday','Prev_1_month_sales', 'Prev_2_month_sales',
                                            'Prev_3_month_sales', 'Prev_4_month_sales'])
    data_df['year'] = data_df['Date'].dt.year
    data_df['month'] = data_df['Date'].dt.month
    data_df['quarter'] = data_df['Date'].dt.quarter
    data_df['Capital_per_labour'] = data_df['Capital'] / data_df['Labour']

    ## Average sales over the past 4 months
    data_df[f"Avg_sales"] = data_df[[f"Prev_{month}_month_sales" for month in range(1,5)]].mean(1)

    ## standard deviation of sales over the past 4 months
    data_df[f"Std_transaction"] = data_df[[f"Prev_{month}_month_sales" for month in range(1,5)]].std(1)

    ## Maximum sales over the past 4 months
    data_df[f"Max_sales"] = data_df[[f"Prev_{month}_month_sales" for month in range(1,5)]].max(1)
    data_df.drop("Date",axis="columns",inplace=True)

    new_df = data_df[['Labour', 'Transactions', 'Capital', 'Store Cluster', 'Holiday', 'year',
       'month', 'quarter', 'Capital_per_labour', 'Prev_1_month_sales',
       'Prev_2_month_sales', 'Prev_3_month_sales', 'Prev_4_month_sales',
       'Avg_sales', 'Std_transaction', 'Max_sales']]

    return new_df
