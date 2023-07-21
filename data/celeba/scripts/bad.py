import pickle
import os 
import random 

num_test_samples = 6000
num_gpair_each_id = 1
num_imppair_each_id = 1

with open('../test/filenames.pickle', 'rb') as f:
    test_files = pickle.load(f)


test_files = sorted(test_files, key = lambda x: int(x.split("/")[0]))
test_files = [i + ".png" for i in test_files]

img_list = [test_files[(i*5):(i*5)+5] for i in range(0, num_test_samples)]
gen_pair = []
imp_pair = []


for i, sub in enumerate(img_list):
    sub = sorted(sub, key = lambda x: int((x.split("_")[1]).split(".")[0]))

    # make geniue pair
    mod_img_list = random.sample(range(1, 5), num_gpair_each_id)
    g_pairs = [sub[0] + " " + sub[mod_img_list[m]] + " 1" for m in range(0, num_gpair_each_id)]
    gen_pair += g_pairs

    # make imposter pair
    mod_sub_list = list(range(0, num_test_samples)) 
    mod_sub_list.pop(i) #removing current subject for making imposter pair

    for j, indx in enumerate(random.sample(mod_sub_list, num_imppair_each_id)):
        imposter_subject = sorted(img_list[indx], key = lambda x: int((x.split("_")[1]).split(".")[0]))
        ip = list(range(1, len(imposter_subject)))
        imp_pair.append(sub[0] + " " + imposter_subject[random.choice(ip)] + " 0")


all_pair = gen_pair + imp_pair
print("gen pair: ", len(gen_pair))
print("imposter pair: ", len(imp_pair))
print("total_pair: ", len(all_pair)) 

with open("celeba_test_2_pair.txt",  'w') as fp:
    for pair in all_pair:
        fp.write("%s\n" % pair)