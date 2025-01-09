# for coco, copy the initial pseudo-labels to semicoco dir
#anno_path="/public/home/lsy/myj/code_dsl/data/semicoco/unlabel_prepared_annos/Industry/annotations/full"
#rm -rf $anno_path
#cp -r workdir_coco/r50_caffe_mslonger_tricks_0.1data/epoch_55.pth-unlabeled.bbox.json_thres0.1_annos/ $anno_path

#dota的开关
# for voc, copy the initial pseudo-labels to semivoc dir
rm -rf ../data/semivoc/unlabel_prepared_annos/Industry/annotations/full10_ca/
#cp -r workdir_voc/RLA_r50_caffe_mslonger_tricks_alldata/epoch_55.pth-unlabeled.bbox.json_thres0.1_annos/ ../data/semivoc/unlabel_prepared_annos/Industry/annotations/full/
cp -r workdir_voc/r50_caffe_mslonger_tricks_07data_built_in/epoch_60.pth-unlabeled.bbox.json_thres0.1_annos/  ../data/semivoc/unlabel_prepared_annos/Industry/annotations/full10_ca/
echo "remove & copy annotations done!"

#CONFIG=configs/fcos_semi/RLA_r50_caffe_mslonger_tricks_0.Xdata_unlabel_dynamic_lw_nofuse_iterlabel_si-soft_singlestage.py
CONFIG=configs/fcos_semi/voc/RLA_r50_caffe_mslonger_tricks_0.Xdata_unlabel_dynamic_lw_nofuse_iterlabel_si-soft_singlestage.py
#WORKDIR=workdir_coco/0.1data
#WORKDIR=workdir_voc/ema_train_richness_hiefilter_0.10data_CA_final_vultra
WORKDIR=workdir_voc/sunzicheng
GPU=2

CUDA_VISIBLE_DEVICES=0,1 PORT=29502 ./tools/dist_train.sh $CONFIG $GPU --work-dir $WORKDIR


#要对nwpu的数据进行dsl训练
# rm -rf /public/home/lsy/myj/code_dsl/NWPU/semivoc/unlabel_prepared_annos/annotations/fullnwpu50
# #cp -r workdir_voc/RLA_r50_caffe_mslonger_tricks_alldata/epoch_55.pth-unlabeled.bbox.json_thres0.1_annos/ ../data/semivoc/unlabel_prepared_annos/Industry/annotations/full/
# cp -r workdir_voc/r50_caffe_mslonger_tricks_50trainnwpu_built_in/epoch_60.pth-unlabeled.bbox.json_thres0.1_annos/  /public/home/lsy/myj/code_dsl/NWPU/semivoc/unlabel_prepared_annos/annotations/fullnwpu50/
# echo "remove & copy annotations done!"
#
# #CONFIG=configs/fcos_semi/RLA_r50_caffe_mslonger_tricks_0.Xdata_unlabel_dynamic_lw_nofuse_iterlabel_si-soft_singlestage.py
# CONFIG=/public/home/lsy/myj/code_dsl/NWPU/RLA_r50_caffe_mslonger_tricks_0.Xdata_unlabel_dynamic_lw_nofuse_iterlabel_si-soft_singlestage.py
# #WORKDIR=workdir_coco/0.1data
# WORKDIR=workdir_voc/ema_train_DSL_50nwpu
# GPU=2
#
# CUDA_VISIBLE_DEVICES=0,1 PORT=29502 ./tools/dist_train.sh $CONFIG $GPU --work-dir $WORKDIR

