import mysql.connector

mydb = mysql.connector.connect(
  host="172.16.163.153",
  user="root",
  password="selab2021",
  database = "test"
)

mycursor = mydb.cursor(prepared=True)

def select_vector(vector_norm):

  # 나중에 주석 코드로 변경 
  #first = vector_norm - 100
  #last = vector_norm + 100 
  
  #adr =  ( first, last )

  sql = "SELECT picture_vector FROM Pictures where picture_price = 12 or token_id = '2343611'"
  #sql = "SELECT picture_vector FROM Pictures"
  #sql = "SELECT picture_vector FROM Pictures WHERE picture_norm BETWEEN %s AND %s"
  
  mycursor.execute(sql)
  #mycursor.execute(sql, adr)

  myresult = mycursor.fetchall()
  final = []

  for x in myresult:
    final.append(x[0])

  return final
  
  

def insert_vector(vector):

  sql = "INSERT INTO Pictures (picture_vector) VALUES (%s)"

  val = "Test"
  # val = vetor
  mycursor.execute(sql, val)

  mydb.commit()

  print(mycursor.rowcount, "record inserted.")
  