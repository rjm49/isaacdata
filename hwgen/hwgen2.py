import random

import pandas as pd
base = "../../../isaac_data_files/"
#choose a teacher for whose class to create homework
teacher_df = pd.read_csv(base+"groups.csv")
groupmem_df = pd.read_csv(base+"group_memberships.csv")

while True:
    r = random.randint(0, teacher_df.shape[0])
    group = teacher_df.iloc[r, :]
    t_id = group["owner_id"]
    g_id = group["id"]
    members = groupmem_df[groupmem_df["group_id"]==g_id]
    member_ids = members["user_id"]
    print(g_id, t_id, list(member_ids))
    input("prompt")