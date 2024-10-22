# Import the library
import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="Mugendo1",
  database = "sentiment"
)


mycursor = mydb.cursor()
mycursor.execute("select * from combined;")
myresult = mycursor.fetchall()
for x in myresult:
  print(x)