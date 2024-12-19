import pandas as pd


def reshape_data(df):

    # Reshape the DataFrame: Pivot years into rows
    df_long = pd.melt(df, id_vars=["Name", "Code", "English short name lower case", "country_iso_2", "country_iso_3"],
                      var_name="Year", value_name="Expenditure")

    # Convert Year to integer
    df_long["Year"] = df_long["Year"].astype(int)

    # Display the reshaped DataFrame
    print(df_long)
    return df_long

def get_output_schema():
    # return pd.DataFrame({
    #      "Name": pd.Series(dtype="str"),
    #     "Code": pd.Series(dtype="str"),
    #     "English short name lower case": pd.Series(dtype="str"),
    #     "country_iso_2": pd.Series(dtype="str"),
    #     "country_iso_3": pd.Series(dtype="str"),
    #     "Year": pd.Series(dtype="int"),
    #     "Expenditure": pd.Series(dtype="float")
    # })

    # return pd.DataFrame({
    #         "Name": ["Sample Name"],
    #         "Code": ["Sample Code"],
    #         "English short name lower case": ["sample name"],
    #         "country_iso_2": ["XX"],
    #         "country_iso_3": ["XXX"],
    #         "Year": [2000],
    #         "Expenditure": [0.0]
    #     })

    return pd.DataFrame({
         "Name": prep_string(),
        "Code": prep_string(),
        "English short name lower case": prep_string(),
        "country_iso_2": prep_string(),
        "country_iso_3": prep_string(),
        "Year": prep_int(),
        "Expenditure": prep_decimal()
    })