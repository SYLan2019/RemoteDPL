import mmcv
import random

# label_data = mmcv.list_from_file('/public/home/lsy/myj/code_dsl/ori_data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt')
# label_len = len(label_data)
#
# unlabel_data = mmcv.list_from_file('/public/home/lsy/myj/code_dsl/data/semivoc/unlabel_prepared_annos/Industry/voc12_trainval.txt')
# unlabel_len = len(unlabel_data)
#
# print('标签数据的数量：', label_len)
# print('未标记标签的数据：', unlabel_len)
# print('标签数据和未标记数据的比例', label_len / unlabel_len)

# train_data_05 = '/public/home/lsy/myj/code_dsl/ori_data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval_1.txt'
# train_data_01 = '/public/home/lsy/myj/code_dsl/ori_data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval_2.txt'
train_data_1 = '/public/home/lsy/myj/code_dsl/ori_data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval_3.txt'

# ratio = 0.1
#
# train1 = int(unlabel_len * ratio)
#
# train1_data = random.sample(label_data, train1)
#
# with open(train_data_1, 'w') as f:
#     for i in train1_data:
#         f.write(i + '\n')
#
# print('done!')

train_list1 = '/public/home/lsy/myj/code_dsl/data/semivoc/prepared_annos/Industry/train_list1.txt'
label_data = mmcv.list_from_file(train_data_1)
with open(train_list1, 'w') as f:
    for i in label_data:
        f.write(i + '.png\n')

