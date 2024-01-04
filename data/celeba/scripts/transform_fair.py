import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import os
from PIL import Image
import random 
import torch 
img_size = 112


def trans(img):
    # resize
    #resize =T.Resize(size=(96, 96))
    #final_resize = T.Resize(size=(img_size, img_size))
    #img = final_resize(resize(img))

    # Random horizontal flipping
    if random.random() > 0.5:
        img = TF.hflip(img)

    # Random rotation
    fill_img = (255, 255, 255)
    angle = T.RandomRotation.get_params(degrees=(-5, 5))
    img = TF.rotate(img, angle, fill=fill_img)

    # random gaussain blur
    blurrer = T.GaussianBlur(kernel_size=(3, 3), sigma=(2, 9))
    img = blurrer(img)

    # random erase
    #erase = T.RandomErasing(p=1.0, scale=(0.35, 0.35), value="random")
    tensor = T.ToTensor()
    img = tensor(img)
    #img = erase(img)

    # color jitter
    jitter = T.ColorJitter(brightness=.4, hue=.5)
    img = jitter(img)

    # random noise
    mean = 0
    std = 0.002**0.5
    noise = torch.normal(mean, std, (3, img_size, img_size))
    img = img + noise 
    return img 


def main():
    src_dir = "./train"
    output_dir = "./trans"
    folder = sorted( os.listdir(src_dir), key = lambda x: int(x.split(".")[0]) )

    for f in folder:
        img_list = os.listdir(os.path.join(src_dir, f))

        for img_file in img_list:
            img_path = os.path.join(src_dir, f, img_file)
            img = Image.open(img_path) 
            trans_img = trans(img)

            os.makedirs(os.path.join(output_dir, f), exist_ok=True)
            save_image(trans_img, os.path.join(output_dir, f,  img_file))

if __name__ == "__main__":
    main()