import mysql.connector
import pandas as pd

OUTPUT_LOCATION = "C:/Users/Gavin/Documents/Gavin/Data Science" \
                  "/M.Sc. Data Science and Analytics/Business Intelligence" \
                  "/Assignments/Assignment 2 - Exploratory Data/python_analysis/"

INPUT_LOCATION = "C:/Users/Gavin/Documents/Gavin/Data Science" \
                 "/M.Sc. Data Science and Analytics/Business Intelligence" \
                 "/Assignments/Assignment 2 - Exploratory Data/datasets/"


def connect_db():
   db = mysql.connector.connect(
        host='localhost',
        port=3306,
        user='root',
        password='Mugendo1'
    )

   return db


def get_data(sql, db):
    cur=db.cursor()
    df = pd.read_sql(sql, db)

    return df



def get_data_info(df):
    """
    Custom function that returns a DataFrame containing DataFrame.info() so that it
    can be output to a file
    :param df:
    :return:
    """
    return pd.DataFrame({"name": df.columns,
                         "non-nulls": len(df) - df.isnull().sum().values,
                         "nulls": df.isnull().sum().values, "type": df.dtypes.values})


def data_out(data, path, separator=','):
    """ Custom function to print list or dict to csv file"""

    if isinstance(data, list) or isinstance(data, dict) or isinstance(data, pd.DataFrame):
        data.to_csv(path_or_buf=path, sep=separator)


def terrorism_dataset_description():
    """
    Custom function to output descriptive information on global
    terrorism dataset
    :return: N/A
    """
    df = pd.read_excel(INPUT_LOCATION + "globalterrorismdb_0522dist.xlsx",
                       sheet_name="Data")

    data_out(df.describe(), OUTPUT_LOCATION + "global_terrorism_describe.csv")
    data_out(get_data_info(df), OUTPUT_LOCATION + "global_terrorism_info.csv")


def military_spend_description():
    """
        Custom function to output descriptive information on global
        military spend dataset
        :return: N/A
        """

    df = pd.read_csv(
        "C:/Users/Gavin/Documents/Gavin/Data Science/M.Sc. Data Science and Analytics/"
        "Business Intelligence/Assignments/Assignment 2 - Exploratory Data/datasets/"
        "Military Expenditure.csv", sep=',')

    data_out(df.describe(), OUTPUT_LOCATION + "military_spend_describe.csv")
    data_out(get_data_info(df), OUTPUT_LOCATION + "military_spend_info.csv")


def iso_country_code_description():
    """
            Custom function to output descriptive information on global
            military spend dataset
            :return: N/A
            """

    df = pd.read_csv(
        "C:/Users/Gavin/Documents/Gavin/Data Science/M.Sc. Data Science and Analytics/"
        "Business Intelligence/Assignments/Assignment 2 - Exploratory Data/datasets/"
        "wikipedia-iso-country-codes.csv", sep=',')

    data_out(df.describe(), OUTPUT_LOCATION + "wikipedia-iso-country-codes_describe.csv")
    data_out(get_data_info(df), OUTPUT_LOCATION + "wikipedia-iso-country-codes_info.csv")


def count_negative_values(df):
    """
    Loops through all numeric attributes in a DataFrame and counts
    how many negative values are in each numeric column.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    pd.DataFrame: A DataFrame with column names and their count of negative values.
    """
    # Select only numeric columns
    numeric_columns = df.select_dtypes(include='number')

    # Count negative values in each numeric column
    negative_counts = numeric_columns.apply(lambda col: (col < 0).sum())

    # Convert the Series to a DataFrame
    negative_counts_df = negative_counts.reset_index()
    negative_counts_df.columns = ['Column', 'Negative_Count']

    return negative_counts_df


# terrorism_dataset_description()
# military_spend_description()
# iso_country_code_description()

# df_gterror = pd.read_excel(
#     "C:/Users/Gavin/Documents/Gavin/Data Science/M.Sc. Data Science and Analytics/"
#     "Business Intelligence/Assignments/Assignment 2 - Exploratory Data/datasets/"
#     "globalterrorismdb_0522dist.xlsx",
#     sheet_name="Data")
#
# result = count_negative_values(df_gterror)
# print(result)
# data_out(result, OUTPUT_LOCATION + "attributes_with_negative_values.csv")

sql = "SELECT * FROM global_terrorism.vw_terrorism_incidents;"
db = connect_db()
df = get_data(sql, db)
# print(df.head())
data_out(get_data_info(df), OUTPUT_LOCATION + "db_view_info.csv")
