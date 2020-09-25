#nohup python train.py --model_type=yolo3_mobilenet_lite --anchors_path=configs/yolo3_anchors.txt --classes_path=configs/my_classes.txt --annotation_file=trainval.txt --val_annotation_file=valid.txt --eval_online --save_eval_checkpoint --total_epoch 300 --batch_size 64 > output.log &

#nohup python train.py --model_type=yolo3_mobilenet_lite_spp --anchors_path=configs/yolo3_anchors.txt --classes_path=configs/my_classes.txt --annotation_file=trainval.txt --val_annotation_file=valid.txt --eval_online --save_eval_checkpoint --total_epoch 300 --batch_size 64 > output.log &

# No data augmentation
#nohup python train.py --model_type=yolo3_mobilenet_lite_spp --anchors_path=configs/yolo3_anchors.txt --classes_path=configs/my_classes.txt --annotation_file=trainval_no_data_aug.txt --val_annotation_file=valid.txt --eval_online --save_eval_checkpoint --total_epoch 300 --batch_size 64 > output.log &

nohup python train.py --model_type=yolo3_darknet --anchors_path=configs/yolo3_anchors.txt --classes_path=configs/my_classes.txt --annotation_file=trainval.txt --val_annotation_file=valid.txt --eval_online --total_epoch 300 --batch_size 64 > output.log &
