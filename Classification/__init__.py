
import data as data

df_adult = data.read_data(file_location="datasets/assesment_adult_dataset.csv", sep=';')
df_student = data.read_data(file_location="datasets/assessment_student_dataset.csv", remove_quotes=True, sep=';\s*')

# clean student data - remove extraneous quotes
df_student = data.clean_data(df_student)

# print data statistics
data.print_data_statistics(df_student)
data.print_data_statistics(df_adult)
data.data_out(df_adult.describe(), "dataout/adult_describe.csv", ",")
data.data_out(df_student.describe(), "dataout/student_describe.csv", ",")


# scale continuous data
data.scale_continuous_data(df_student)
data.scale_continuous_data(df_adult)
data.print_data_statistics(df_student)
data.print_data_statistics(df_adult)
data.data_out(df_adult.describe(), "dataout/adult_describe_scaled.csv", ",")
data.data_out(df_student.describe(), "dataout/student_describe_scaled.csv", ",")


# validate
