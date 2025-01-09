# 说明
## 各个方法对应环境配置

实验机器配置：8cpu核心 2张3090 NVIDIA-SMI 470.74 	Driver Version: 470.74	CUDA Version: 11.4

**RemoteDPL**：

pytorch                   1.10.1          py3.9_cuda11.3_cudnn8.2.0_0 

torchvision               0.11.2               py39_cu113

mmcv                      1.3.10                   pypi_0 

mmcv-full                 1.4.0                    pypi_0

python                    3.9.16               h7a1cb2a_1 


## 数据说明

### DOTAv1.0数据集：

​	•	**图像总数**：2806 张。

​	•	**图像大小**：800 × 800 ～ 4000 × 4000 像素（高分辨率）。

​	•	**目标总数**：超过 188,282 个标注。

​	•	**类别数量**：15 个目标类别。

​	DOTA v1.0 定义了以下 15 个类别的目标：

​	1.	**Plane**（飞机）

​	2.	**Baseball diamond**（棒球场）

​	3.	**Bridge**（桥梁）

​	4.	**Ground track field**（跑道）

​	5.	**Small vehicle**（小型车辆）

​	6.	**Large vehicle**（大型车辆）

​	7.	**Ship**（船只）

​	8.	**Tennis court**（网球场）

​	9.	**Basketball court**（篮球场）

​	10.	**Storage tank**（储罐）

​	11.	**Soccer-ball field**（足球场）

​	12.	**Roundabout**（环岛）

​	13.	**Harbor**（港口）

​	14.	**Swimming pool**（游泳池）									

​	15.	**Helicopter**（直升机）


### NWPU数据集

​        •	**VHR-10**：800 张图像。

​	•	**图像大小**：512 × 512 像素。

​	•	**目标总数**：超过 3000 个目标实例。

​	•	**类别数量**：10 个目标类别。

​	NWPU 数据集包含以下 10 个目标类别：

​	1.	**Airplane**（飞机）

​	2.	**Ship**（船只）

​	3.	**Storage tank**（储罐）

​	4.	**Baseball diamond**（棒球场）

​	5.	**Tennis court**（网球场）

​	6.	**Basketball court**（篮球场）

​	7.	**Ground track field**（跑道）

​	8.	**Harbor**（港口）

​	9.	**Bridge**（桥梁）

​	10.	**Vehicle**（车辆，包括小型和大型车辆）



## RemoteDPL实验内容

### 训练步骤
0.按照DSL的readme创建数据所在文件夹

1、转换coco数据为semicoco数据

``````pyt
cd DSL（打开dsl所在文件夹）
python3 tools/coco_convert2_semicoco_json.py --input ${project_root_dir}/ori_data/coco --output ${project_root_dir}/data/semicoco
``````

2、训练supervised baseline model

```````python
./demo/model_train/baseline_voc.sh
```````

3、产生初始伪标签

``````python
./tools/inference_unlabeled_coco_data.sh
``````

4、转换为DSL风格的注释

``````python
python3 tools/generate_unlabel_annos_coco.py --input_path /public/home/lsy/myj/code_dsl/DSL/workdir_voc/r50_caffe_mslonger_tricks_50trainnwpu_built_in/epoch_60.pth-unlabeled.bbox.json --cat_info /public/home/lsy/myj/code_dsl/NWPU/mmdet_category_info.json --thres 0.1
``````

5、进行DSL训练

``````pyt
./demo/model_train/unlabel_dynamic.sh
``````

### RemoteDPL训练的细节——参数修改

发生于以上步骤5

切换RemoteDPL训练需要：

- 1 配置文件.l\DSL\configs\fcos_semi\voc\RLA_r50_caffe_mslonger_tricks_0.Xdata_unlabel_dynamic_lw_nofuse_iterlabel_si-soft_singlestage.py中，设置bbox_head的type为FCOSHead_richness，使用代用density分支的头
- 2 依旧在配置文件打开unlabel pred的richness属性，这是为了在生成的伪标签标记中具有boost属性，用来进行二次过滤
- 3 依旧在配置文件中，down_fuse参数修改为true，用于控制生成下采样特征图
- 4 code_dsl/DSL/mmdet/runner/hooks/unlabel_pred_hook.py:295这一行，需要将down_fuse修改为true，从而走伪标签生产阶段-即二次过滤阶段
- 5 code_dsl/DSL/mmdet/models/detectors/single_stage.py:152打开density损失-即richess损失，取消注释就行

**PS:**

仅打开1即对应消融实验使用了density 头——验证dendity

仅打开3对应多尺度联合训练——验证多尺度联合训练

打开1和3表示对应多尺度联合训练和densitt分支——验证分阶段过滤

打开全部表示总模型

code_dsl/DSL/mmdet/models/detectors/single_stage.py:114开启表示使用图5c也就是最好的融合模块

code_dsl/DSL/mmdet/models/detectors/single_stage.py:117开启表示图5b对应的模块

code_dsl/DSL/mmdet/models/detectors/single_stage.py:119-122表示图5a对应的模块

code_dsl/DSL/mmdet/datasets/semivoc.py:74：设置stage ming中的第二阶段过滤滤值：\(\sigmoid_3\)

code_dsl/DSL/mmdet/models/dense_heads/fcos_head_rich.py:339：设置密集度估计分支损失的权重即\(\beta\)

训练步骤5中：关闭1、2、3、4、5并且关闭配置文件code_dsl/DSL/configs/fcos_semi/voc/RLA_r50_caffe_mslonger_tricks_0.Xdata_unlabel_dynamic_lw_nofuse_iterlabel_si-soft_singlestage.py:258 
对应生成图三a**单分支**的结果
训练步骤5中，关闭1、2、3、4、5并且开启配置文件code_dsl/DSL/configs/fcos_semi/voc/RLA_r50_caffe_mslonger_tricks_0.Xdata_unlabel_dynamic_lw_nofuse_iterlabel_si-soft_singlestage.py:258 对应图3b**双分支**的结果

打卡3，关闭code_dsl/DSL/configs/fcos_semi/voc/RLA_r50_caffe_mslonger_tricks_0.Xdata_unlabel_dynamic_lw_nofuse_iterlabel_si-soft_singlestage.py:258 对应图三c**三分支**的结果

### 日志

RemoteDPL日志文件在/public/home/lsy/myj/code_dsl/DSL/workdir_voc

#