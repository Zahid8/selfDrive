import pathlib
import datetime
import glob
from pathlib import Path

import os
# os.chdir('/home/rafal/PycharmProjects/naukaOpenCV/saves')
#
# print(len(list(Path('.').glob('*')))) ## 1
# print(len(list(Path('.').glob('**/*')))) ## 2

classes = list(range(-20, 21))
# print(classes)
# print(datetime.datetime.now().time())



def file_counter(e):
    return len(os.listdir(f'/home/rafal/PycharmProjects/naukaOpenCV/new_map_saves1.1/{e}'))

def folders_counter():
    return len(os.listdir(f'/home/rafal/PycharmProjects/naukaOpenCV/new_map_saves1.1'))


for c in classes:
    # os.chdir(f'/home/rafal/PycharmProjects/naukaOpenCV/saves/{c}')
    # # pathlib.Path(f'/home/rafal/PycharmProjects/naukaOpenCV/saves/{c}').mkdir(parents=True, exist_ok=True)
    # print(f'{c}: {len(list(Path(".").glob("*")))}')  ## 1
    # print(f'/home/rafal/PycharmProjects/naukaOpenCV/saves/{c}')
    # lista = os.listdir(f'/home/rafal/PycharmProjects/naukaOpenCV/saves/{c}')  # dir is your directory path
    number_files = len(os.listdir(f'/home/rafal/PycharmProjects/naukaOpenCV/new_map_saves1.1/{c}'))
    # if number_files < 1000:
    print(f'{c}: {number_files}')
    # print(file_counter(c))

counter = 0
# while True:
#     if all(file_counter(c) for c in classes) == 100:
#         print('done')
#     else:
#         print('not done')

