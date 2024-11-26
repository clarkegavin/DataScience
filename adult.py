import data as data  # custom utility
import plots as plts  # custom utility


class Adult:
    def __init__(self):
        self.df = None


    def process_data(self, df):

        self.df = data.read_data(file_location="datasets/assesment_adult_dataset.csv", sep=';')

        # print data statistics
        data.data_out(self.df.describe(), "dataout/adult_describe_numerical.csv", ",")
        data.data_out(self.df.describe(include='object'), "dataout/adult_describe_categorical.csv", ",")
        data.data_out(data.get_data_info(self.df), "dataout/adult_info.csv")

        # scale continuous data
        df_scaled, scaler = data.scale_continuous_data(self.df)
        # output scaled data description for validation
        data.data_out(df_scaled.describe(), "dataout/adult_describe_scaled.csv", ",")

        # Separate the descriptive attributes from the class label column
        df_descriptive, df_predictive = data.remove_class_label(self.df, 'label')  # unscaled
        df_descriptive_scaled, df_predictive_scaled = data.remove_class_label(df_scaled,
                                                                                          'label')  # scaled

        # print(f"Distribution of adult dataset class label: {df_adult['label'].value_counts()}")

        # Encode adult categorical data using one hot encoder
        df_encoded, encoder = data.encode_categorical_data_ohe(df_descriptive)  # unscaled/encoded
        df_adult_encoded_scaled, encoder = data.encode_categorical_data_ohe(
            df_descriptive_scaled)  # scaled/encoded
        data.data_out(df_encoded, "dataout/adult_ohe.csv")
        data.data_out(data.get_data_info(df_encoded), "dataout/adult_ohe_info.csv")

        # plts.histogram(df_adult, "Adult data")
        # plts.histogram(df_scaled, "Adult data scaled")

        # Check correlation

        #df_adult_class_encoded = self.df['label'].apply(lambda x: 1 if x == '>50K' else 0)
        # Note: Calculating and displaying scatter plots using onehotencoder is unrealistic given the 'pivot' that OHE creates
        # df_adult_correlation = pd.concat([df_encoded, df_adult_class_encoded], axis='columns')
        # print(f"Fully encoded: {df_adult_correlation.head()}")
        # print(f"Encoded description:\n {df_adult_correlation.info()}")
        plts.scatter_plot_matrix(self.df, hue='label')
        #plts.plot_correlation_matrix(df_encoded)

        return self.df, df_encoded, scaler


