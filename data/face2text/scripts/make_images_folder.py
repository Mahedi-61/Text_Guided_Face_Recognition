import os 
import shutil 

src = "../images_org"
img_files = os.listdir(src)
id_files = {} 


with open("identity_CelebA.txt", "r") as f:
    lines = f.readlines()


for l in lines:
    key = l.split(" ")[0]
    value = (l.split(" ")[1])[:-1]
    id_files[key] = value

for img in img_files:
    org_id = id_files[img]
    print(org_id)

    dst_dir = os.path.join("images", str(org_id))
    os.makedirs(dst_dir, exist_ok=True)
    
    dst_path = os.path.join(dst_dir, img)
    src_path = os.path.join(src, img)
    
    print(dst_path)
    shutil.move(src_path, dst_path)

