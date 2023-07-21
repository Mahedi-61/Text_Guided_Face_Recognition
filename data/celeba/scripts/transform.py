import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import os
from PIL import Image
import random 
import torch 
img_size = 152

def only_resize(img):
    final_resize = T.Resize(size=(img_size, img_size))
    img = final_resize(img)

    tensor = T.ToTensor()
    img = tensor(img)
    return img 


def trans(img):
    # resize
    resize =T.Resize(size=(112, 112))
    final_resize = T.Resize(size=(img_size, img_size))
    img = final_resize(resize(img))

    # Random horizontal flipping
    if random.random() > 0.5:
        img = TF.hflip(img)

    # Random rotation
    fill_img = (255, 255, 255)
    angle = T.RandomRotation.get_params(degrees=(-40, 40))
    img = TF.rotate(img, angle, fill=fill_img)

    # random gaussain blur
    blurrer = T.GaussianBlur(kernel_size=(5, 5), sigma=(2, 8))
    img = blurrer(img)

    
    # random erase
    #erase = T.RandomErasing(p=1.0, scale=(0.35, 0.35), value="random")
    tensor = T.ToTensor()
    img = tensor(img)
    #img = erase(img)

    # color jitter
    jitter = T.ColorJitter(brightness=.4, hue=.3)
    img = jitter(img)

    # random noise
    mean = 0
    std = 0.005**0.5
    noise = torch.normal(mean, std, (3, img_size, img_size))
    img = img + noise 
    return img 


def main():
    num_trans = 5
    folder = sorted( os.listdir("../images_org"), key = lambda x: int(x.split(".")[0]))
    img_list = [os.path.join("../images_org", i) for i in folder]

    for img_path in img_list[:24000]:
        img = Image.open(img_path) 

        for i in range(num_trans):
            if i == 9: label = "_10"
            else: label = "_0" + str(i + 1)

            #if i == 0: trans_img = only_resize(img)
            #else: trans_img = trans(img)
            trans_img = trans(img)

            img_id = (img_path.split('/')[-1]).split('.')[0]
            output_dir = os.path.join("train_images_DAMSM", img_id)
            os.makedirs(output_dir, exist_ok=True)
            save_image(trans_img, os.path.join(output_dir, img_id + label + ".png"))



if __name__ == "__main__":
    main()