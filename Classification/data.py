import pandas as pd
import csv

from sklearn.preprocessing import StandardScaler


def read_data(file_location, remove_quotes=False, sep=','):
    """
        custom function to read data from flat files, removing quotation qualifiers if necessary
    """
    if remove_quotes:
        df = pd.read_csv(file_location, sep=sep, skipinitialspace=True, quoting=csv.QUOTE_ALL, engine='python')
    else:
        df = pd.read_csv(file_location, sep=';')

    return df


def clean_data(df):
    # remove extraneous quotes
    df.columns = df.columns.str.replace('"', '')
    df = df.map(lambda x: x.replace('"', '') if isinstance(x, str) else x)
    return df


def print_data_statistics(df):
    """ Custom function to print dataframe statistics to console"""
    df.info()
    print(df.describe())


def data_out(data, path, separator=','):
    """ Custom function to print list or dict to csv file"""
    if isinstance(data, list) or isinstance(data, dict) or isinstance(data, pd.DataFrame):
        data.to_csv(path_or_buf=path, sep=separator)


def scale_continuous_data(df):
    """ Custom function to scale numerical data"""
    numeric_cols = df.select_dtypes(include='float64').columns
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[numeric_cols])

    df[numeric_cols] = scaled_values
    return df

def encode_categorical_data(df):
    # TODO: Complete this function
    """ Custom function to encode categorical data"""
    pass