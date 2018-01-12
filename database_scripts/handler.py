'''
Created on 30 Mar 2017

@author: Russell
'''

#!/usr/bin/python

import psycopg2
import json
from _sqlite3 import OperationalError

with open('db_config.json') as json_data_file:
    config = json.load(json_data_file)
    db_config = config["postgresql"]
print(db_config)

conn = psycopg2.connect(database=db_config["db"], user=db_config["user"], password=db_config["passwd"], host=db_config["host"], port=db_config["port"])
c = conn.cursor()
print("Opened database successfully")

fd = open('query_all.sql', 'r')
sqlFile = fd.read()
fd.close()

print(sqlFile)
# all SQL commands (split on ';')
sqlCommands = sqlFile.split(';')

# Execute every command from the input file
for command in sqlCommands:
    print("->",command)
    # This will skip and report errors
    # For example, if the tables do not yet exist, this will skip over
    # the DROP TABLE commands
    try:
        c.execute(command)
    except OperationalError as msg:
        print("Command skipped: ", msg)

    col_names = [i[0] for i in c.description]

    f = open("isaac_MC_data_all", "w")
    f.write(", ".join(col_names) +"\n")
    for rec in c:
        print(rec)
        f.write(repr(rec)+"\n")
    f.close()