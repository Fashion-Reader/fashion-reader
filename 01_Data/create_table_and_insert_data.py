
import pymysql


from pymongo import MongoClient
from tqdm import tqdm

client = MongoClient("mongodb+srv://sun:0000@crawling-data.r9oex.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db = client["crawling-data"]["imvely"]

data = [i for i in db.find()]

# CONNECT DB
db = pymysql.connect(host='database.cawpd3yaf0pl.us-east-2.rds.amazonaws.com',
                     user='admin',
                     passwd='00000000')
cursor = db.cursor()

# USE DATABASE
SQL = """
use products_table
"""
cursor.execute(SQL)

# DROP TABLE
SQL = """
DROP TABLE IF EXISTS products
"""
cursor.execute(SQL)

# CREATE TABLE
SQL = """
CREATE TABLE IF NOT EXISTS products (
 item_id INT,
 name VARCHAR(100),
 url VARCHAR(200),
 item_type VARCHAR(30),  
 item_type_id INT,  
 price VARCHAR(10),
 size_options VARCHAR(100),
 color_options VARCHAR(100),
 item_img_links VARCHAR(500),
 thickness VARCHAR(30),
 see_through VARCHAR(30),
 flexibility VARCHAR(30),
 touch VARCHAR(30),
 fit VARCHAR(30),
 lining VARCHAR(30), 
 PRIMARY KEY(item_id)
) ENGINE= MYISAM CHARSET=utf8
"""
#  shoulder_width VARCHAR(10),
#  chest_size VARCHAR(10),
#  armhole VARCHAR(10),
#  sleeve_length VARCHAR(10),
#  sleeve_size VARCHAR(10),
cursor.execute(SQL)

# CHECK TABLE
SQL = """
select count(*) from products
"""
cursor.execute(SQL)
print(cursor.fetchone())

# INSERT DATA
for i in tqdm(range(len(data))):
    try:
        SQL = """
        INSERT INTO 
            products(item_id, name, url, item_type, item_type_id, price, size_options, color_options, item_img_links, thickness, see_through, flexibility, touch, fit, lining)
            values('%d', '%s', '%s', '%s', '%d', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')
        """ % (int(data[i]["item_id"]), data[i]["name"], data[i]["url"], data[i]["item_type"], 
               int(data[i]["item_type_id"]), data[i]["price"].replace(",", ""), data[i]["size_options"], 
               data[i]["color_options"], data[i]["item_img_links"], data[i]["thickness"],
               data[i]["see_through"], data[i]["flexibility"], data[i]["touch"], data[i]["fit"], data[i]["lining"])
        cursor.execute(SQL)
        db.commit()
    except:
        continue


# CHECK TABLE
SQL = """
select count(*) from products
"""
cursor.execute(SQL)
print(cursor.fetchall())
