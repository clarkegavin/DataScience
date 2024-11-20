import pandas as pd
import csv
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ExceptionHandling import CustomFileNotFoundError


def read_data(file_location, remove_quotes=False, sep=','):
    """
        custom function to read data from flat files, removing quotation qualifiers if necessary
    """
    try:
        if remove_quotes:
            df = pd.read_csv(file_location, sep=sep, skipinitialspace=True, quoting=csv.QUOTE_ALL, engine='python')
        else:
            df = pd.read_csv(file_location, sep=';')
        return df
    except FileNotFoundError:
        raise CustomFileNotFoundError(file_name=file_location)


def clean_data(df):
    # remove extraneous quotes
    df.columns = df.columns.str.replace('"', '')
    df = df.map(lambda x: x.replace('"', '') if isinstance(x, str) else x)
    return df


def print_data_statistics(df):
    """ Custom function to print dataframe statistics to console"""
    df.info()
    print(df.describe())


def get_data_info(df):
    """
    Custom function that returns a DataFrame containing DataFrame.info() so that it can be output to a file
    :param df:
    :return:
    """
    return pd.DataFrame({"name": df.columns, "non-nulls": len(df) - df.isnull().sum().values,
                         "nulls": df.isnull().sum().values, "type": df.dtypes.values})


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


def encode_categorical_data_le(df):
    """ Custom function to encode categorical data and return the encoded dataframe and a dictonary
        with the original mappings
    """
    label_encoders = {}
    for column in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le  # Store encoders if need to reverse transformation

    return df, label_encoders


def encode_categorical_data_ohe(df):
    pass
