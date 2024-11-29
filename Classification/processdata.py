import os

import Classification.data as data  # custom utility
import Classification.plots as plts  # custom utility


class ProcessData:
    def __init__(self, df, data_out="dataout/"):
        self.df = df
        #self.data_in = data_in
        self.data_out = data_out

    def get_dataframe(self):
        return self.df

    def clean_data(self):
        data.clean_data(self.df)

    def process_data(self):

        # print data statistics
        try:
            data.data_out(self.df.describe(), self.data_out + "describe_numerical.csv", ",")
            data.data_out(self.df.describe(include='object'), self.data_out + "describe_categorical.csv", ",")
            data.data_out(data.get_data_info(self.df), self.data_out + "info.csv")
        except OSError as e:
            os.makedirs(self.data_out)
            data.data_out(self.df.describe(), self.data_out + "describe_numerical.csv", ",")
            data.data_out(self.df.describe(include='object'), self.data_out + "describe_categorical.csv", ",")
            data.data_out(data.get_data_info(self.df), self.data_out + "info.csv")


        # plts.histogram(self.df, "Adult data")

        # Check correlation

        # df_adult_class_encoded = self.df['label'].apply(lambda x: 1 if x == '>50K' else 0)
        # Note: Calculating and displaying scatter plots using onehotencoder is unrealistic given the 'pivot' that OHE creates
        # df_adult_correlation = pd.concat([df_encoded, df_adult_class_encoded], axis='columns')
        # print(f"Fully encoded: {df_adult_correlation.head()}")
        # print(f"Encoded description:\n {df_adult_correlation.info()}")
        # plts.scatter_plot_matrix(self.df, hue='label')
        # plts.plot_correlation_matrix(df_encoded)

        return self.df

    def plot_correlation_matrix(self, class_label):
        plts.scatter_plot_matrix(self.df, hue=class_label)

    def scale_data(self, df, type = 'Standard'):
        # scale continuous data
        if type == 'Standard':
            df_scaled, scaler = data.scale_continuous_data(df)
            # output scaled data description for validation
        elif type == 'MinMaxScaler':
            df_scaled, scaler = data.scale_min_max(df)

        data.data_out(df_scaled.describe(), self.data_out + "describe_scaled.csv", ",")
        return df_scaled, scaler

    def encode_data(self, df, class_label, encoder='ohe'):
        # Separate the descriptive attributes from the class label column
        df_descriptive, df_predictive = data.remove_class_label(df, class_label)

        if encoder == 'ohe':
            # Encode adult categorical data using one hot encoder
            df_encoded, encoder = data.encode_categorical_data_ohe(df_descriptive)  # unscaled/encoded
        else:
            df_encoded, encoder = data.encode_categorical_data_le(df_descriptive)

        data.data_out(df_encoded, self.data_out + "encoded.csv")
        data.data_out(data.get_data_info(df_encoded), self.data_out + "encoded_info.csv")
        return df_encoded, df_predictive, encoder
