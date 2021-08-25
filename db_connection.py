import mysql.connector

mydb = mysql.connector.connect(
  host="172.16.163.153",
  user="root",
  password="selab2021",
  database = "test"
)

mycursor = mydb.cursor(prepared=True)

def select_vector(vector_norm):

  first = vector_norm - 20
  last = vector_norm + 20 
  adr =  ( first, last )

  sql = "SELECT picture_vector FROM Pictures WHERE picture_norm BETWEEN %s AND %s"
  
  mycursor.execute(sql, adr)

  myresult = mycursor.fetchall()

  for x in myresult:
    print(x)

  return myresult
  
  

def insert_vector(vector):

  sql = "INSERT INTO Pictures (picture_vector) VALUES (%s)"

  val = "Test"
  # val = vetor
  mycursor.execute(sql, val)

  mydb.commit()

  print(mycursor.rowcount, "record inserted.")
  