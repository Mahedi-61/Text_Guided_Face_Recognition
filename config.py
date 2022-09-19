class Config(object):
    dataset = "birds"
    
    if dataset == "webface":
        backbone = 'resnet18'
        classify = 'softmax'
        num_classes = 13938 
        metric =   "arc_margin" 
        easy_margin = False
        loss =  "focal_loss" 
        optimizer = "sgd" 
        use_se = False

        train_root = "./data/Datasets/webface/CASIA-maxpy-clean/"
        train_list = "./data/Datasets/webface/cleaned_list.txt"
        test_root = "./data/Datasets/lfw/lfw_align-128"                  
        test_pair_list = "./data/Datasets/lfw/lfw_test_pair.txt"     

    elif dataset == "birds":
        backbone = 'resnet18'
        classify = 'softmax'
        num_classes = 200 
        metric =   "nothing" 
        easy_margin = False
        loss =  "cross_entropy" 
        optimizer = "adam" 
        use_se = False

        train_root = "./data/birds/CUB_200_2011/images"  
        train_list = "./data/birds/CUB_200_2011/cleaned_list.txt"  
        test_root = "./data/birds/CUB_200_2011/images"                  
        test_pair_list = "./data/birds/CUB_200_2011/birds_test_pair.txt"     


    ####################### common ######################
    env = 'default'
    display = False
    finetune = False

    checkpoints_path = './checkpoints'
    load_model_path = './weights/resnet18_birds_245.pth'
    save_interval = 5
    test_interval = 5

    train_batch_size = 256  # batch size
    test_batch_size = 128
    input_shape = (1, 128, 128)
    
    use_gpu = True  # use GPU or not
    gpu_id = '0,1'
    num_workers = 2  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 250
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4