import pandas as pd
import csv
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from ExceptionHandling import CustomFileNotFoundError


def read_data(file_location, remove_quotes=False, sep=','):
    """
        custom function to read data from flat files, removing quotation qualifiers if necessary
    """
    try:
        if remove_quotes:
            #df = pd.read_csv(file_location, sep=sep, skipinitialspace=True, quoting=csv.QUOTE_ALL, engine='python')
            df = pd.read_csv(file_location, sep=sep, skipinitialspace=True, quotechar='"', engine='python')
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
    # print(df.describe(include='object'))
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
    df_copy = df.copy()  # scale a copy, otherwise the original dataframe gets altered
    numeric_cols = df_copy.select_dtypes(include='float64').columns
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df_copy[numeric_cols])

    df_copy[numeric_cols] = scaled_values
    return df_copy, scaler

def scale_min_max(df):
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include='number').columns
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df_copy[numeric_cols])
    df_copy[numeric_cols] = scaled_values
    return df_copy, scaler


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
    """
    Custom function to encode categorical data using one hot encoder
    :param df: Input Dataframe with categorical columns to encode
    :return: Encoded DataFrame and a dictionary of OneHotEncoder objects
    """

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_array = ohe.fit_transform(df[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=ohe.get_feature_names_out(categorical_cols),
        index=df.index
    )
    non_categorical_df = df.drop(columns=categorical_cols)
    final_df = pd.concat([non_categorical_df, encoded_df], axis=1)

    return final_df, ohe


def remove_class_label(df, class_label):
    df_descriptive = df[df.columns.difference([class_label])]
    df_predictive = df[[class_label]]
    return df_descriptive, df_predictive


def get_numerical_attributes(df):
    numerical_cols = df.select_dtypes(include=['object', 'category']).columns
    return numerical_cols