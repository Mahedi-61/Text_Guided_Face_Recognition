import os 
import shutil 

src = "../text_org"
img_files = os.listdir(src)
raw_files = {}
id_files = {} 


with open("mapping.txt", "r") as f:
    lines = f.readlines()

lines = lines[1:]
for l in lines:
    key = l.split(" ")[0]
    value = (l.split(".")[0]).split(" ")[-1] + ".jpg"
    raw_files[key] = value

with open("identity_CelebA.txt", "r") as f:
    lines = f.readlines()


for l in lines:
    key = l.split(" ")[0]
    value = (l.split(" ")[1])[:-1]
    id_files[key] = value


for img in img_files:
    org_file = raw_files[img.split(".")[0]]
    org_id = id_files[org_file]

    dst_dir = os.path.join("text", str(org_id))
    os.makedirs(dst_dir, exist_ok=True)
    
    dst_path = os.path.join(dst_dir, img)
    src_path = os.path.join(src, img)
    
    print(dst_path)
    shutil.move(src_path, dst_path)
