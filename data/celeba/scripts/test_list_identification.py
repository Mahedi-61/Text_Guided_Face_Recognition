import pickle
import os 
import random 

num_genuine_per_sub = 1
imposter_subject = 400 - 1 #1 genuine and 199 impsoter
set = "test"
img_dir = "../images/%s" % set
all_subjects = sorted(os.listdir(img_dir), key = lambda x: int(x))

ls_subjectes = [sorted(os.listdir(os.path.join(img_dir, sub)), 
                key= lambda x: int((x.split(".")[0]).split("_")[-1])) 
                for sub in all_subjects]

imp_pairs = []
all_pairs = []

for k in range(num_genuine_per_sub):
    """
    for each subject we have one genuine pair 
    and 'imposter subject' imposter pair
    This loop will continue num_genuine_per_sub times 
    """

    for i, sub_imgs in enumerate(ls_subjectes):
        # make geniue pair
        selected_img_list = list(range(0, len(sub_imgs))) #k+1 to not pic genuine image
        if len(selected_img_list) > 1: selected_img_list.pop(k)
        g_pairs = [sub_imgs[k] + " " + sub_imgs[selected_img_list[0]] + " 1"]

        # make imposter pair
        selected_sub_list = ls_subjectes.copy() 
        selected_sub_list.pop(i) #removing current subject for making imposter pair
        
        imp_pairs = [sub_imgs[k] + " " + sel_sub[random.choice(range(0, len(sel_sub)))] + " 0" 
                    for sel_sub in selected_sub_list[:imposter_subject] ]
        
        all_pairs +=  g_pairs + imp_pairs

print("creating txt files")
with open("celeba_%s_%d_sub.txt" % (set, imposter_subject),  'w') as fp:
    for pair in all_pairs:
        fp.write("%s\n" % pair)