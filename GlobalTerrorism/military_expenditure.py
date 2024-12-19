import pandas as pd

def fill_missing_values(dataframe):
    """
    Fills null values in the dataset with the previous
    year's value for the same country.

    Parameters:
    dataframe (pd.DataFrame): A DataFrame
    where rows represent years , columns represent country, and values are their annual spend.

    Returns:
    pd.DataFrame: The DataFrame with missing values filled.
    """
    # Iterate through the DataFrame and fill missing values
    # Iterate through the DataFrame and fill missing values
    # Identify year columns (assumes year columns are integers)
    year_columns = [col for col in dataframe.columns
                    if str(col).isdigit() and 1960 <= int(col) <= 2020]

    # Sort the year columns to ensure correct chronological order
    year_columns.sort(key=lambda x: int(x))

    print(year_columns)
    # Iterate through the DataFrame and fill missing values for year columns only
    for country in dataframe.index:
        for i in range(1, len(year_columns)):
            year = year_columns[i]
            prev_year = year_columns[i - 1]
            if pd.isnull(dataframe.at[country, year]):
                dataframe.at[country, year] = dataframe.at[country, prev_year]

    return dataframe
