import mysql.connector

mydb = mysql.connector.connect(
  host="172.16.163.153",
  user="root",
  password="selab2021"
)

print(mydb)