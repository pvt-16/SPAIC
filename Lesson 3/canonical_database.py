# -*- coding: utf-8 -*-
"""canonical database.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vmXAf36tTkfftRBtu9SkNDbo2VpRW5KU
"""

import torch

number_of_entries = 5000
db = torch.rand(number_of_entries)> 0.5

db.shape

db[0:6]

#generate parallel databases - one entry removed

removed_index = 2

new_db = torch.cat((db[0:removed_index],db[removed_index+1:]))

new_db
#new_db.shape

def get_parallel_database(db,removed_index):
  
  return torch.cat((db[0:removed_index],db[removed_index+1:]))

def get_x_number_parallel_databases(db):
  
  databases = list()
  for i in range(len(db)):
    databases.append(get_parallel_database(db,i))
    
  return databases

def create_db_and_parallels(number_of_entries):
  db = torch.rand(number_of_entries)> 0.5
  parallel_dbs = get_x_number_parallel_databases(db)
  
  return db,parallel_dbs

db, pdbs = create_db_and_parallels(20)

db

pdbs

