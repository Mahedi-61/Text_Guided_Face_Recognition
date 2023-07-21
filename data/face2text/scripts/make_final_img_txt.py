import os 
import shutil 
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import os
from PIL import Image
import random 
import torch 
img_size = 112

def only_resize(img):
    final_resize = T.Resize(size=(img_size, img_size))
    img = final_resize(img)

    tensor = T.ToTensor()
    img = tensor(img)
    return img 


def trans(img):
    # resize
    final_resize = T.Resize(size=(img_size, img_size))
    img = final_resize(img)

    # Random horizontal flipping
    if random.random() > 0.5:
        img = TF.hflip(img)

    # Random rotation
    fill_img = (255, 255, 255)
    angle = T.RandomRotation.get_params(degrees=(-10, 10))
    img = TF.rotate(img, angle, fill=fill_img)
    
    tensor = T.ToTensor()
    img = tensor(img)
    return img 


def saving_images(trans_img, new_i, count):
    label = str(new_i) + "_" + str(count)
    output_dir = os.path.join("images_final", str(new_i))
    os.makedirs(output_dir, exist_ok=True)

    dst_dir = os.path.join(output_dir, label + ".jpg")
    save_image(trans_img, dst_dir)


def saving_texts(text_path, new_i, count):
    label = str(new_i) + "_" + str(count)
    output_dir = os.path.join("text_final", str(new_i))
    os.makedirs(output_dir, exist_ok=True)

    dst_dir = os.path.join(output_dir, label + ".txt")
    shutil.copy(text_path, dst_dir)

# loading text and image data
text_folders = os.listdir("text_org_id")
img_folders = os.listdir("images_org_id")

# sorted 
text_folders = sorted(text_folders, key= lambda x : int(x))
img_folders = sorted(img_folders, key= lambda x : int(x)) 

# matching where the text and image folder have the same ids & same images
assert text_folders == img_folders; print("matched") 


# at least 3 images & text per subject 
# copy from the src dir; so running following this code is safe
for new_i, id in enumerate(img_folders):
    iid_dir = os.path.join("images_org_id", id)
    tid_dir = os.path.join("text_org_id", id)
    num_of_imges = len(os.listdir(iid_dir))

    count = 0
    if  num_of_imges < 3: 
        diff = 3 - num_of_imges
    else: diff = 0

    for img_name in os.listdir(iid_dir):
        img_path = os.path.join(iid_dir, img_name)
        img = Image.open(img_path) 
        trans_img = only_resize(img)
        count += 1
        saving_images(trans_img, new_i, count)
        text_path = os.path.join(tid_dir, img_name.replace(".jpg", ".txt"))
        saving_texts(text_path, new_i, count)

    for d in range(diff):
        trans_img = trans(img)
        count += 1
        saving_images(trans_img, new_i, count)
        text_path = os.path.join(tid_dir, img_name.replace(".jpg", ".txt"))
        saving_texts(text_path, new_i, count)
        
    if (new_i % 1000 == 0): print(new_i) 
