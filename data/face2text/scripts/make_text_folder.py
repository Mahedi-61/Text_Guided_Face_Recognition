import os 
import shutil 

src = "../text"
folders = os.listdir(src)
id_files = {} 


with open("identity_CelebA.txt", "r") as f:
    lines = f.readlines()


for l in lines:
    key = l.split(" ")[0]
    value = (l.split(" ")[1])[:-1]
    id_files[key] = value


for text_folder in folders:
    text_file = text_folder + "_01.txt"
    org_id = id_files[text_folder + ".jpg"]

    dst_dir = os.path.join("text_org_id", str(org_id))
    os.makedirs(dst_dir, exist_ok=True)
    
    dst_path = os.path.join(dst_dir, text_folder + ".txt")
    src_path = os.path.join(src, text_folder, text_file)
    
    print(dst_path)
    shutil.move(src_path, dst_path)
