import json
import os 
import shutil 
import random
from collections import Counter

dict_caption = dict()

with open('raw_2.0.jsonl', 'r') as f:
    array = f.readlines()

for i in range(len(array)):
    d = json.loads(array[i])
    key = d["filename"]

    value = d["description"]

    if key in dict_caption:
        dict_caption[key].append(value)
    else:
        dict_caption[key] = [value]


for key in dict_caption.keys():
    src_dir = os.path.join("../img_align_celeba", key)
    dst_dir = os.path.join("../images_org", key)
    shutil.copy(src_dir, dst_dir)


"""
files = "../images_org"
for f in os.listdir(files):
    dst_dir = os.path.join("../images", f.split(".")[0])
    os.makedirs(dst_dir, exist_ok=True)

    dst_path = os.path.join(dst_dir, f)
    src_path = os.path.join(files, f)
    shutil.move(src_path, dst_path)
"""


def make_texts(dict_caption):
    for k in dict_caption.keys():
        text_file =  k.split(".")[0]
        text_folder = os.path.join("../text", text_file)
        os.makedirs(text_folder, exist_ok=True) 

        with open(os.path.join(text_folder, text_file + "_01.txt"), "w") as file:
            ls_caption = dict_caption[k]
            if len(ls_caption) == 1:
                file.write(ls_caption[0] + "\n")
                file.write(ls_caption[0] + "\n")
                file.write(ls_caption[0] + "\n")
                file.write(ls_caption[0])
            
            if len(ls_caption) == 2:
                file.write(ls_caption[0] + "\n")
                file.write(ls_caption[1] + "\n")
                file.write(ls_caption[0] + "\n")
                file.write(ls_caption[1])

            if len(ls_caption) == 3:
                file.write(ls_caption[0] + "\n")
                file.write(ls_caption[1] + "\n")
                file.write(ls_caption[2] + "\n")
                file.write(ls_caption[random.choice([0, 1, 2])])
            
            if len(ls_caption) > 3:
                [file.write(ls_caption[i] + "\n") if i < 3 else file.write(ls_caption[i]) for i in range(0, 4)]

    for f in os.listdir("../text"):
        src_txt = os.path.join("../text", f, f+"_01.txt")
        dst_txt = os.path.join("../text", f, f)
        a = [shutil.copy(src_txt, dst_txt+i) for i in ["_02.txt", "_03.txt", "_04.txt", "_05.txt"]]
