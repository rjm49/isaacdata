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

with open(dir_path+'/db_config.json') as json_data_file:
    config = json.load(json_data_file)
    db_config = config["postgresql"]
print(db_config)

conn = psycopg2.connect(database=db_config["db"], user=db_config["user"], password=db_config["passwd"], host=db_config["host"], port=db_config["port"])
c = conn.cursor()
print("Opened database successfully")


# all SQL commands (split on ';')
sqlCommands = [
    # ("groups","select * from groups"),
    # ("users","select * from users"),
    # ("all_pids","select distinct page_id from content_data"),
    # ("role_changes", "select * from logged_events where event_type='CHANGE_USER_ROLE'"),
    ("user_preferences", "select * from user_preferences"),
    ("teachers", """"select users.id, family_name, given_name, email, preference_type, preference_name, preference_value from users, user_preferences where users.role='TEACHER' and users.id=user_id
and preference_type in ('EMAIL_PREFERENCE','BETA_FEATURE') and preference_value=TRUE
order by family_name, given_name""")
]

for command_pair in sqlCommands:
    print("command pair",command_pair)
    fname = command_pair[0]
    command = command_pair[1]
    try:
        c.execute(command)
    except OperationalError as msg:
        print("Command skipped: ", msg)

    print("received {} rows".format(c.rowcount))
    col_names = [i[0] for i in c.description]
    df = pandas.DataFrame(c.fetchall(), columns = col_names)
    df.to_csv(fname+".csv", index=None)

