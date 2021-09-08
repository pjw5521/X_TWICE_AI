import mysql.connector

def select_vector(vector_norm):

  mydb = mysql.connector.connect(
  host="172.16.163.153",
  user="root",
  password="selab2021",
  database = "test"
  )

  mycursor = mydb.cursor(prepared=True)
  # 나중에 주석 코드로 변경 
  first = vector_norm - 100
  last = vector_norm + 100 
  
  adr =  ( first, last )

  #sql = "SELECT picture_vector FROM Pictures where token_id = '2343444' or token_id = '2343504'"
  sql = "SELECT picture_vector FROM Pictures where Not token_id is null"
  #sql = "SELECT picture_vector FROM Pictures WHERE picture_norm BETWEEN %s AND %s"
  
  mycursor.execute(sql)
  #mycursor.execute(sql, adr)

  myresult = mycursor.fetchall()
  final = []

  for x in myresult:
    final.append(x[0])

  #print("final", final)

  mydb.close()
  
  return final
  

def getTokenId(vector):

  mydb = mysql.connector.connect(
  host="172.16.163.153",
  user="root",
  password="selab2021",
  database = "test"
  )

  mycursor = mydb.cursor(prepared=True)
  adr = ( vector, )
  sql ="select picture_url From Pictures WHERE picture_vector = %s"
  mycursor.execute(sql, adr)

  myresult = mycursor.fetchall()
  
  result = myresult[0][0]
  
  mydb.close()

  return result
