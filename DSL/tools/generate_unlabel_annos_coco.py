import json
import os
import argparse

import mmcv

import sys
sys.path.append('/public/home/lsy/myj/code_dsl/DSL')
from mmdet.datasets.api_wrappers import COCO

# CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
#                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
#                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
#                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
#                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
#                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
# dota数据集
# CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
#             'small-vehicle', 'large-vehicle', 'ship',
#             'tennis-court', 'basketball-court',
#             'storage-tank', 'soccer-ball-field',
#             'roundabout', 'harbor',
#             'swimming-pool', 'helicopter')
# nwpu数据集
CLASSES = ('airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court',
           'basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle')


def report(args, dst_path):
    #coco = COCO("data/coco_semi/semi_supervised/instances_train2017.json")

    # 下面暂时注释掉
    # coco = COCO(args.input_list)
    # cat_ids = coco.get_cat_ids(cat_names=CLASSES)
    # cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}

    #myj 2023.6.4 需要重新写一下cat_ids和种类到标签的映射
    cat_ids = [i + 1 for i in range(len(CLASSES))]
    cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    label2word = json.load(open(args.cat_info,'r'))['id2cat']
    cnt=0
    with open(args.input_path,'r') as f:
        data = json.load(f)
        #{'image_id': 391895, 'bbox': [346.72308349609375, 19.586660385131836, 125.62673950195312, 280.1312656402588], 'score': 0.8345049619674683, 'category_id': 1}
        for i in range(len(data)):
            #name = coco.load_imgs([data[i]['image_id']])[0]['file_name'] + '.json'
            #png图片用下面的，dota
            # name = data[i]['image_id'] + '.png.json'
            name = data[i]['image_id'] + '.jpg.json'

            if not os.path.exists(os.path.join(dst_path, name)):
                with open(os.path.join(dst_path, name),'w') as F:
                    tmp_data={}
                    #tmp_data["imageName"] = os.path.join("full/", coco.load_imgs([data[i]['image_id']])[0]['file_name'])
                    #myj 2023.6.5 重写
                    # tmp_data['imageName'] = os.path.join('full/', data[i]['image_id'] + '.png')
                    tmp_data['imageName'] = os.path.join('full/', data[i]['image_id'] + '.jpg')
                    tmp_data["targetNum"] = 0
                    tmp_data["rects"] = []
                    tmp_data["tags"] = []
                    tmp_data["scores"] = []
                    tmp_data["masks"] = []
                    json.dump(tmp_data,F,ensure_ascii=False,indent=4)
            if float(data[i]['score']) <= float(args.thres):
                continue
            cnt +=1
            if cnt%10000 ==0:
                print(cnt)
            if os.path.exists(os.path.join(dst_path, name)):
                with open(os.path.join(dst_path, name),'r') as F:
                    tmp_data = json.load(F)
                tmp_data["targetNum"] = tmp_data["targetNum"] + 1
                tmp_data["rects"].append([data[i]['bbox'][0], data[i]['bbox'][1], data[i]['bbox'][0]+data[i]['bbox'][2], data[i]['bbox'][1]+data[i]['bbox'][3]])
                tmp_data["tags"].append(label2word[str(cat2label[data[i]['category_id']])])
                tmp_data["scores"].append(data[i]['score'])
                tmp_data["masks"].append([])
                with open(os.path.join(dst_path,name),'w') as F:
                    json.dump(tmp_data,F,ensure_ascii=False,indent=4)
    print("total num : ",cnt)

    cnt = 0
    #myj 2023.6.5 对于没有检测出物体的图像，也得为他们生成空白伪标签
    #with open(args.input_list,'r') as f:
        #data=json.load(f)
    print('读入文件列表', args.input_list)
    data = mmcv.list_from_file(args.input_list)
    # for i in range(len(data['images'])):
    #     if not os.path.exists(os.path.join(dst_path, data['images'][i]['file_name']+'.json')):
    #         with open(os.path.join(dst_path, data['images'][i]['file_name']+'.json'),'w') as F:
    #             cnt +=1
    #             tmp_data={}
    #             tmp_data["imageName"] = os.path.join("full/", data['images'][i]['file_name'])
    #             tmp_data["targetNum"] = 0
    #             tmp_data["rects"] = []
    #             tmp_data["tags"] = []
    #             tmp_data["scores"] = []
    #             tmp_data["masks"] = []
    #             json.dump(tmp_data,F,ensure_ascii=False,indent=4)

    # myj 2023.6.5 重写
    for i in data:
        if not os.path.exists(os.path.join(dst_path, i + '.json')):
            with open(os.path.join(dst_path, i + '.json'), 'w') as f:
                cnt += 1
                tmp_dic = {}
                tmp_dic['imageName'] = os.path.join("full/", i)
                tmp_dic["targetNum"] = 0
                tmp_dic["rects"] = []
                tmp_dic["tags"] = []
                tmp_dic["scores"] = []
                tmp_dic["masks"] = []
                json.dump(tmp_dic, f, ensure_ascii=False,indent=4)

    print("没有伪标签的图片数目为: ", cnt)
                    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="setting input path and output path")
    parser.add_argument('--input_path',type=str,default=None,help="path to XX-unlabel.bbox.json")
    # parser.add_argument('--input_list',type=str,default='/public/home/lsy/myj/code_dsl/data/semivoc/unlabel_prepared_annos/Industry/voc12_trainval.txt',help="path to data list: data_list/coco_semi/semi_supervised/instances_train2017.2@10-unlabeled.json")
    parser.add_argument('--input_list', type=str,
                        default='/public/home/lsy/myj/code_dsl/NWPU/unlabel.txt',
                        help="path to data list: data_list/coco_semi/semi_supervised/instances_train2017.2@10-unlabeled.json")
    parser.add_argument('--cat_info',type=str,default=None,help="path to category_info: /gruntdata1/bhchen/factory/data/semicoco/mmdet_category_info.json")
    #处理掉预测score得分低于threshold的目标实例
    parser.add_argument('--thres',type=float,default=0.1,help="threshold: 0.1 is used as default")
    args= parser.parse_args()
    dst_path = os.path.abspath(args.input_path) + '_thres'+str(args.thres)+'_annos/'
    if not os.path.exists(dst_path):
            os.makedirs(dst_path)
    report(args, dst_path)
