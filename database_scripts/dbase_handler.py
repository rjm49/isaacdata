'''
Created on 30 Mar 2017

@author: Russell
'''

#!/usr/bin/python
import pandas
import psycopg2
import json
from _sqlite3 import OperationalError
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

def get_group_deets(userlist):
    with open(dir_path+'/db_config.json') as json_data_file:
        config = json.load(json_data_file)
        db_config = config["postgresql"]
    print(db_config)

    conn = psycopg2.connect(database=db_config["db"], user=db_config["user"], password=db_config["passwd"], host=db_config["host"], port=db_config["port"])
    c = conn.cursor()
    print("Opened database successfully")

    fd = open(dir_path+'/userlist_query.sql', 'r')
    sqlFile = fd.read()
    fd.close()

    # all SQL commands (split on ';')
    sqlCommands = sqlFile.split(';')

    df_list = []
    # Execute every command from the input file
    for command in sqlCommands:
        print(command)
        # This will skip and report errors
        # For example, if the tables do not yet exist, this will skip over
        # the DROP TABLE commands

        userlist = map(str, userlist)
        userlist = ["'"+x+"'" for x in userlist]
        stru = ",".join(userlist)
        print(stru)
        try:
            c.execute(command.replace("%s",stru))
        except OperationalError as msg:
            print("Command skipped: ", msg)

        col_names = [i[0] for i in c.description]
        df = pandas.DataFrame(c.fetchall(), columns = col_names)

        # f = open("isaac_MC_data_all", "w")
        # f.write(",".join(col_names) +"\n")

        print("received {} rows".format(c.rowcount))
        # for ix, rec in enumerate(c):
        #     df.loc[ix] = rec
            #print("added row")
        print("built df")
        # f.close()
        df_list.append(df)
    return df_list