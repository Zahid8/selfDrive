# import csv
# import os.path
# from pathlib import Path
# counter = 0
#
# with open('labels.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',')
#     for row in reader:
#         if len(row)>0:
#             # print(row[0])
#             img = row[0]
#             my_file = Path(f"/home/rafal/PycharmProjects/naukaOpenCV/saves_full_copy/{img}")
#             # print(my_file)
#             if my_file.is_file():
#                 print(img)
#                 counter += 1
#             # counter +=1
#             # print(row[0])
#             # if os.path.isfile(f'/home/rafal/PycharmProjects/naukaOpenCV/saves_full_copy/{row[0]}') == 1:
#             #     print('file_found')
#
#
# with open('labels_left.csv', newline='') as inp, open('labels_left _edited.csv', mode='a') as out:
#     writer = csv.writer(out, delimiter=',')
#     reader = csv.reader(inp, delimiter=',')
#     for row in reader:
#         if len(row)>0:
#             img = row[0]
#             my_file = Path(f"/home/rafal/PycharmProjects/naukaOpenCV/saves_full_copy/{img}")
#             if my_file.is_file():
#                 writer.writerow(row)
#
# print(counter)

import os
rootdir = '/home/rafal/PycharmProjects/naukaOpenCV/people'

counter = 0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        # old_file = os.path.join("directory", "a.txt")
        # new_file = os.path.join("directory", "b.kml")
        os.rename(os.path.join(subdir, file), os.path.join(subdir, f'people{counter}.png'))
        counter += 1