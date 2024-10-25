
import mysql.connector
import pandas as pd

mydb = mysql.connector.connect(host="127.0.0.1", user="root", password="Mugendo1", database="tbi")
mycursor = mydb.cursor(buffered=True)
mycursor.execute("""select distinct country from tbi.customer order by country desc;""")
df = pd.DataFrame(mycursor.fetchall(), columns=['country'])
print(df)

mycursor.execute("""select * from tbi.customer;""")
columns = [desc[0] for desc in mycursor.description]
df_customer = pd.DataFrame(mycursor.fetchall(), columns=columns)
print(df_customer)
print(df_customer.describe())
