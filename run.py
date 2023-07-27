import os
import argparse
import logging
import sys
sys.path.append("..")

import torch
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from RADF.models.models import RADFREModel, RADFNERModel
from RADF.models.bert_model import HMNeTREModel, HMNeTNERModel
from processor.dataset import MMREProcessor, MMPNERProcessor, MMREDataset, MMPNERDataset
from modules.train import RETrainer, NERTrainer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# from tensorboardX import SummaryWriter

import wandb
import time

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


MODEL_CLASSES = {
    'MRE': RADFREModel,
    'twitter15': RADFNERModel,
    'twitter17': RADFNERModel
}

TRAINER_CLASSES = {
    'MRE': RETrainer,
    'twitter15': NERTrainer,
    'twitter17': NERTrainer
}
DATA_PROCESS = {
    'MRE': (MMREProcessor, MMREDataset),
    'twitter15': (MMPNERProcessor, MMPNERDataset), 
    'twitter17': (MMPNERProcessor, MMPNERDataset)
}

DATA_PATH = {
    'MRE': {
            # text data
            'train': 'data/RE_data/txt/ours_train.txt',    
            'dev': 'data/RE_data/txt/ours_val.txt',
            'test': 'data/RE_data/txt/ours_test.txt',
            # {data_id : object_crop_img_path}
            'train_auximgs': 'data/RE_data/txt/mre_train_dict.pth',
            'dev_auximgs': 'data/RE_data/txt/mre_dev_dict.pth',
            'test_auximgs': 'data/RE_data/txt/mre_test_dict.pth',
            # relation json data
            're_path': 'data/RE_data/ours_rel2id.json',
            # image feature path
            'train_imgfeas':'data/RE_data/img_obj_10/img_train_10.pickle',
            'dev_imgfeas': 'data/RE_data/img_obj_10/img_val_10.pickle' ,
            'test_imgfeas': 'data/RE_data/img_obj_10/img_test_10.pickle'

            },
    
    'twitter15': {
                # text data
                'train': 'data/NER_data/twitter2015/train.txt',
                'dev': 'data/NER_data/twitter2015/valid.txt',
                'test': 'data/NER_data/twitter2015/test.txt',
                # {data_id : object_crop_img_path}
                'train_auximgs': 'data/NER_data/twitter2015/twitter2015_train_dict.pth',
                'dev_auximgs': 'data/NER_data/twitter2015/twitter2015_val_dict.pth',
                'test_auximgs': 'data/NER_data/twitter2015/twitter2015_test_dict.pth'
            },

    'twitter17': {
                # text data
                'train': 'data/NER_data/twitter2017/train.txt',
                'dev': 'data/NER_data/twitter2017/valid.txt',
                'test': 'data/NER_data/twitter2017/test.txt',
                # {data_id : object_crop_img_path}
                'train_auximgs': 'data/NER_data/twitter2017/twitter2017_train_dict.pth',
                'dev_auximgs': 'data/NER_data/twitter2017/twitter2017_val_dict.pth',
                'test_auximgs': 'data/NER_data/twitter2017/twitter2017_test_dict.pth'
            },
        
}

# image data
IMG_PATH = {
    'MRE': {'train': 'data/RE_data/img_org/train/',
            'dev': 'data/RE_data/img_org/val/',
            'test': 'data/RE_data/img_org/test'},
    'twitter15': 'data/NER_data/twitter2015_images',
    'twitter17': 'data/NER_data/twitter2017_images',
}

# auxiliary images
AUX_PATH = {
    'MRE':{
            'train': 'data/RE_data/img_vg/train/crops',
            'dev': 'data/RE_data/img_vg/val/crops',
            'test': 'data/RE_data/img_vg/test/crops'
    },
    'twitter15': {
                'train': 'data/NER_data/twitter2015_aux_images/train/crops',
                'dev': 'data/NER_data/twitter2015_aux_images/val/crops',
                'test': 'data/NER_data/twitter2015_aux_images/test/crops',
            },

    'twitter17': {
                'train': 'data/NER_data/twitter2017_aux_images/train/crops',
                'dev': 'data/NER_data/twitter2017_aux_images/val/crops',
                'test': 'data/NER_data/twitter2017_aux_images/test/crops',
            }
}

# object-level feature
IMG_OBJ_PATH = {
    'MRE':{
            'train': 'data/RE_data/img_obj/train/',
            'dev': 'data/RE_data/img_obj/val/',
            'test': 'data/RE_data/img_obj/test/'
    },
    'twitter15': {
                'train': 'data/NER_data/twitter2015_aux_images/train/',
                'dev': 'data/NER_data/twitter2015_aux_images/val/',
                'test': 'data/NER_data/twitter2015_aux_images/test/',
            },

    'twitter17': {
                'train': 'data/NER_data/twitter2017_obj/train/',
                'dev': 'data/NER_data/twitter2017_obj/val/',
                'test': 'data/NER_data/twitter2017_obj/test/',
            }
}


def set_seed(seed=2021):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='twitter15', type=str, help="The name of dataset.")
    parser.add_argument('--bert_name', default='bert-base-uncased', type=str, help="Pretrained language model path")
    parser.add_argument('--num_epochs', default=30, type=int, help="num training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--lr', default=0.00001, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=16, type=int, help="epoch to start evluate")
    parser.add_argument('--seed', default=1, type=int, help="random seed, default is 1")
    parser.add_argument('--prompt_len', default=10, type=int, help="prompt length")
    parser.add_argument('--prompt_dim', default=800, type=int, help="mid dimension of prompt project layer")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default=None, type=str, help="save model at save_path")
    parser.add_argument('--write_path', default=None, type=str, help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--use_prompt', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--max_seq', default=128, type=int)
    parser.add_argument('--ignore_idx', default=-100, type=int)
    parser.add_argument('--sample_ratio', default=1.0, type=float, help="only for low resource.")
    # SGR
    parser.add_argument('--sim_size', default=256, type=int, help="Dimensionality of the similarity embedding.") 
    parser.add_argument('--sgr_step', default=3, type=int, help="The number of the Graph Reasoning Step.") 
    # DIME
    parser.add_argument('--num_head_FSRU', type=int, default=16, help='Number of heads in Feature Semantic Reasoning Unit')
    parser.add_argument('--hid_FSRU', type=int, default=512, help='Hidden size of FeedForward in Feature Semantic Reasoning Unit')
    parser.add_argument('--raw_feature_norm_CMRC', default="clipped_l2norm", help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--lambda_softmax_CMRC', default=4., type=float, help='Attention softmax temperature.')
    parser.add_argument('--hid_router', type=int, default=512, help='Hidden size of MLP in routers')
    parser.add_argument('--embed_size', default=256, type=int, help='Dimensionality of the joint embedding.')
    # Visual Net
    parser.add_argument('--img_dim', default=2048, type=int, help='Dimensionality of the image embedding.')
    parser.add_argument('--finetune', action='store_true', help='Fine-tune the image encoder.')
    parser.add_argument('--cnn_type', default='vgg19', help="""The CNN used for image encoder(e.g. vgg19, resnet152)""")
    parser.add_argument('--use_abs', action='store_true', help='Take the absolute value of embedding vectors.') 
    parser.add_argument('--no_imgnorm', action='store_true', help='Do not normalize the image embeddings.')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout')

    args = parser.parse_args()

    data_path, img_path, aux_path  = DATA_PATH[args.dataset_name], IMG_PATH[args.dataset_name], AUX_PATH[args.dataset_name] 
    img_objfea = IMG_OBJ_PATH[args.dataset_name] 
    model_class, Trainer = MODEL_CLASSES[args.dataset_name], TRAINER_CLASSES[args.dataset_name]
    data_process, dataset_class = DATA_PROCESS[args.dataset_name]
    '''
    data_path: {'train': 'data/RE_data/txt/ours_train.txt', 'dev': 'data/RE_data/txt/ours_val.txt', 'test': 'data/RE_data/txt/ours_test.txt', 'train_auximgs': 'data/RE_data/txt/mre_train_dict.pth', 'dev_auximgs': 'data/RE_data/txt/mre_dev_dict.pth', 'test_auximgs': 'data/RE_data/txt/mre_test_dict.pth', 're_path': 'data/RE_data/ours_rel2id.json'}
    img_path : {'train': 'data/RE_data/img_org/train/', 'dev': 'data/RE_data/img_org/val/', 'test': 'data/RE_data/img_org/test'}
    aux_path : {'train': 'data/RE_data/img_vg/train/crops', 'dev': 'data/RE_data/img_vg/val/crops', 'test': 'data/RE_data/img_vg/test/crops'}
    model_class: <class 'models.bert_model.HMNeTREModel'>
    Trainer : <class 'modules.train.RETrainer'>
    data_process :<class 'processor.dataset.MMREProcessor'>
    dataset_class : <class 'processor.dataset.MMREDataset'>
    '''
    
    transform = transforms.Compose([
        transforms.Resize(256),#改变大小
        transforms.CenterCrop(224),#中心裁剪
        transforms.ToTensor(),#转化为矩阵
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])#使用Imagenet的均值和标准差是一种常见的做法。 它们是根据数百万张图像计算得出的。 如果要在自己的数据集上从头开始训练，则可以计算新的均值和标准差。 否则，建议使用Imagenet预设模型及其平均值和标准差。

    set_seed(args.seed) # set seed, default is 1
    if args.save_path is not None:  # make save_path dir
        # args.save_path = os.path.join(args.save_path, args.dataset_name+"_"+str(args.batch_size)+"_"+str(args.lr)+"_"+args.notes)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)


    print(args)

    logdir = "logs/" + args.dataset_name+ "_"+str(args.batch_size) + "_" + str(args.lr) + args.notes
    # writer = SummaryWriter(logdir=logdir)
    # writer=None


    ti = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%Y-%m-%d_%H:%M:%S',
                    level = logging.INFO)
    logger = logging.getLogger(__name__)

    writer = wandb.init(project='RADF',
           name=args.dataset_name+"_"+ti,
           config=args,
           resume='allow')

    processor = data_process(data_path, args.bert_name)#<processor.dataset.MMREProcessor object at 0x7effb32043a0>
    train_dataset = dataset_class(processor, transform, img_path, aux_path, img_objfea, args.max_seq, sample_ratio=args.sample_ratio, mode='train')#<processor.dataset.MMREDataset object at 0x7eff45369af0>

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)#<torch.utils.data.dataloader.DataLoader object at 0x7eff45369ac0>

    dev_dataset = dataset_class(processor, transform, img_path, aux_path, img_objfea, args.max_seq, mode='dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=False)

    test_dataset = dataset_class(processor, transform, img_path, aux_path, img_objfea, args.max_seq, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=False)

    if args.dataset_name == 'MRE':  # RE task
        re_dict = processor.get_relation_dict()
        num_labels = len(re_dict)#{'None': 0, '/per/per/parent': 1, '/per/per/siblings': 2, '/per/per/couple': 3, '/per/per/neighbor': 4, '/per/per/peer': 5, '/per/per/charges': 6, '/per/per/alumi': 7, '/per/per/alternate_names': 8, '/per/org/member_of': 9, '/per/loc/place_of_residence': 10, '/per/loc/place_of_birth': 11, '/org/org/alternate_names': 12, '/org/org/subsidiary': 13, '/org/loc/locate_at': 14, '/loc/loc/contain': 15, '/per/misc/present_in': 16, '/per/misc/awarded': 17, '/per/misc/race': 18, '/per/misc/religion': 19, '/per/misc/nationality': 20, '/misc/misc/part_of': 21, '/misc/loc/held_on': 22}
        tokenizer = processor.tokenizer
        model = RADFREModel(num_labels, tokenizer, args=args)
        trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model, processor=processor, args=args, logger=logger, writer=writer)
    else:   # NER task
        label_mapping = processor.get_label_mapping()
        label_list = list(label_mapping.keys())
        model = HMNeTNERModel(label_list, args)

        trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model, label_map=label_mapping, args=args, logger=logger, writer=writer)

    if args.do_train:
        # train
        trainer.train()
        # test best model
        args.load_path = os.path.join(args.save_path, 'best_model.pth')
        trainer.test()

    if args.only_test:
        # only do test
        trainer.test()

    torch.cuda.empty_cache()
    # writer.close()
    

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()