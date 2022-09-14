class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 13938
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False

    train_root = './data/Datasets/webface/CASIA-maxpy-clean/'
    train_list = './data/Datasets/webface/cleaned_list.txt'
    #train_list = './data/Datasets/webface/train_data_13938.txt'
    #val_list = './data/Datasets/webface/val_data_13938.txt'


    lfw_root = "./data/Datasets/celeba"                   #lfw/lfw-align-128'
    lfw_test_list = "./data/Datasets/celeba/celeba_test_pair.txt"     #lfw/lfw_test_pair.txt'

    checkpoints_path = './checkpoints'
    load_model_path = './weights/resnet18_110.pth'
    test_model_path = './checkpoints/resnet18_95.pth'
    save_interval = 5
    test_interval = 5

    train_batch_size = 256  # batch size
    test_batch_size = 128

    input_shape = (1, 128, 128)
    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 2  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 150
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
