import os 
import pickle 

folder_path = "text_final"
folders = sorted(os.listdir(folder_path), key = lambda x: int(x))
train_filenames = []
valid_filenames = []
test_filenames = []

train_cls_id = []
valid_cls_id = []
test_cls_id = []

num_test_samples = 1193
num_valid_samples = 500
num_train_samples = 4500

for f in folders[:num_train_samples]:
    cls_id = int(f) 
    
    cls_txt = os.listdir(os.path.join(folder_path, f))
    cls_txt = [os.path.join(f, i[:-4]) for i in cls_txt]
    cls_txt = sorted(cls_txt, key=lambda x: int(x.split("_")[-1]))

    train_cls_id += [cls_id] * len(cls_txt)
    train_filenames += cls_txt


for f in folders[num_train_samples: (num_train_samples + num_valid_samples)]:
    cls_id = int(f) 
    
    cls_txt = os.listdir(os.path.join(folder_path, f))
    cls_txt = [os.path.join(f, i[:-4]) for i in cls_txt]
    cls_txt = sorted(cls_txt, key=lambda x: int(x.split("_")[-1]))

    valid_cls_id += [cls_id] * len(cls_txt)
    valid_filenames += cls_txt


start = num_train_samples + num_valid_samples
for f in folders[start: ]:
    cls_id = int(f) 
    
    cls_txt = os.listdir(os.path.join(folder_path, f))
    cls_txt = [os.path.join(f, i[:-4]) for i in cls_txt]
    cls_txt = sorted(cls_txt, key=lambda x: int(x.split("_")[-1]))

    test_cls_id += [cls_id] * len(cls_txt)
    test_filenames += cls_txt


list_filepath = ["train_filenames.pickle", "valid_filenames.pickle", "test_filenames.pickle"]
list_files = [train_filenames, valid_filenames, test_filenames]

for file, filepath in zip(list_files, list_filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(file, f)


list_filepath = ["train_class_info.pickle", "valid_class_info.pickle", "test_class_info.pickle"]
list_files = [train_cls_id, valid_cls_id, test_cls_id]

for file, filepath in zip(list_files, list_filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(file, f)