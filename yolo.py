#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a YOLOv3/YOLOv2 style detection model on test images.
"""

import colorsys
import os, sys, argparse
import cv2
import time
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Lambda
from tensorflow_model_optimization.sparsity import keras as sparsity
from PIL import Image, ImageFile

from utils import constraints, dendrometrics, segmentation
from yolo3.model import get_yolo3_model, get_yolo3_inference_model#, get_yolo3_prenms_model
from yolo3.postprocess_np import yolo3_postprocess_np
from yolo2.model import get_yolo2_model, get_yolo2_inference_model
from yolo2.postprocess_np import yolo2_postprocess_np
from common.data_utils import preprocess_image
from common.utils import get_classes, get_anchors, get_colors, draw_boxes
from tensorflow.keras.utils import multi_gpu_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#tf.enable_eager_execution()

default_config = {
        "model_type": 'tiny_yolo3_darknet',
        "weights_path": os.path.join('weights', 'yolov3-tiny.h5'),
        "pruning_model": False,
        "anchors_path": os.path.join('configs', 'tiny_yolo3_anchors.txt'),
        "classes_path": os.path.join('configs', 'coco_classes.txt'),
        "score" : 0.1,
        "iou" : 0.4,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }


class YOLO_np(object):
    _defaults = default_config

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        super(YOLO_np, self).__init__()
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = get_classes(self.classes_path)
        self.anchors = get_anchors(self.anchors_path)
        self.colors = get_colors(self.class_names)
        K.set_learning_phase(0)
        self.yolo_model = self._generate_model()

    def _generate_model(self):
        '''to generate the bounding boxes'''
        weights_path = os.path.expanduser(self.weights_path)
        assert weights_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        #YOLOv3 model has 9 anchors and 3 feature layers but
        #Tiny YOLOv3 model has 6 anchors and 2 feature layers,
        #so we can calculate feature layers number to get model type
        num_feature_layers = num_anchors//3

        try:
            if num_anchors == 5:
                # YOLOv2 use 5 anchors
                yolo_model, _ = get_yolo2_model(self.model_type, num_anchors, num_classes, input_shape=self.model_image_size + (3,), model_pruning=self.pruning_model)
            else:
                yolo_model, _ = get_yolo3_model(self.model_type, num_feature_layers, num_anchors, num_classes, input_shape=self.model_image_size + (3,), model_pruning=self.pruning_model)
            yolo_model.load_weights(weights_path) # make sure model, anchors and classes match
            if self.pruning_model:
                yolo_model = sparsity.strip_pruning(yolo_model)
            yolo_model.summary()
        except Exception as e:
            print(repr(e))
            assert yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
        print('{} model, anchors, and classes loaded.'.format(weights_path))
        if self.gpu_num>=2:
            yolo_model = multi_gpu_model(yolo_model, gpus=self.gpu_num)

        return yolo_model


    def detect_image(self, image, apply_constraints=False):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'

        image_data = preprocess_image(image, self.model_image_size)
        #origin image shape, in (height, width) format
        image_shape = tuple(reversed(image.size))

        start = time.time()
        out_boxes, out_classes, out_scores = self.predict(image_data, image_shape)
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        end = time.time()
        print("Inference time: {:.8f}s".format(end - start))
        
        start_c = 0
        end_c = 0
        if (apply_constraints):
            start_c = time.time()            
            print('Applying constraints...')
            out_boxes, out_classes, out_scores = constraints.apply_constraints(self.class_names, 
                                                                               out_boxes, 
                                                                               out_classes, 
                                                                               out_scores)            
            end_c = time.time()
            print("Constraints time: {:.8f}s".format(end_c - start_c))
            
        if (not os.path.exists('results')):
            os.mkdir('results')
            
        time_file = open('results/time.txt', 'a')                        
        time_file.write('{}\n'.format([end - start, end_c - start_c]))
        time_file.close()
            

        #draw bounding boxes on input image
        image_array = np.array(image, dtype='uint8')
        image_array = draw_boxes(image_array, out_boxes, out_classes, out_scores, self.class_names, self.colors)
        return Image.fromarray(image_array), [out_boxes, out_classes, self.class_names]


    def predict(self, image_data, image_shape):
        num_anchors = len(self.anchors)
        if num_anchors == 5:
            # YOLOv2 use 5 anchors
            out_boxes, out_classes, out_scores = yolo2_postprocess_np(self.yolo_model.predict(image_data), image_shape, self.anchors, len(self.class_names), self.model_image_size, max_boxes=100)
        else:
            out_boxes, out_classes, out_scores = yolo3_postprocess_np(self.yolo_model.predict(image_data), image_shape, self.anchors, len(self.class_names), self.model_image_size, max_boxes=100)
        return out_boxes, out_classes, out_scores


    def dump_model_file(self, output_model_file):
        self.yolo_model.save(output_model_file)
        
    def overlap_rate(self, bbox1, bbox2):
        width = bbox1[2] - bbox1[0]
        height = bbox1[-1] - bbox1[1]
        
        condition = list()
        
        for x in range(bbox2[0], bbox2[0] + (bbox2[2] - bbox2[0])):
            for y in range(bbox2[1], bbox2[1] + (bbox2[-1] - bbox2[1])):
                condition.append(bbox1[0] <= x < (bbox1[0]+ width) and bbox1[1] <= y < (bbox1[1] + height))
        
        n = (bbox2[2] - bbox2[0]) * (bbox2[-1] - bbox2[1])
        condition = np.array(condition)
        
        return sum(condition) / n
    
    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        
        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        
        # return the intersection over union value
        return iou    
    
    def remove_duplicate_boxes(self, bbox1_ind, bbox2, out_boxes, out_classes, out_scores, method='overlap'):
        if (method!='overlap' and method!='iou'):
            raise Exception('Remove duplicate boxes: method type must be \'overlap\' or \'iou\'!')
        
        larger_overlap = -9999
        larger_overlap_index = -1
        
        indices_to_remove = list()
        
        if (method=='overlap'):
            for i in bbox1_ind:
                # Calculating the overlapping between the ith element of the bbox1 and the bbox2
                overlap = self.overlap_rate(out_boxes[i], bbox2[0])
                
                if (overlap > larger_overlap):
                    larger_overlap = overlap
                    
                    if (larger_overlap_index >= 0):
                        indices_to_remove.append(larger_overlap_index)
                    
                    larger_overlap_index = i
                else:
                    indices_to_remove.append(i)
        else:
            for i in bbox1_ind:
                # Calculating the overlapping between the ith element of the bbox1 and the bbox2
                overlap = self.bb_intersection_over_union(out_boxes[i], bbox2[0])
                
                if (overlap > larger_overlap):
                    larger_overlap = overlap
                    
                    if (larger_overlap_index >= 0):
                        indices_to_remove.append(larger_overlap_index)
                    
                    larger_overlap_index = i
                else:
                    indices_to_remove.append(i)            
        
        out_boxes = np.delete(out_boxes,indices_to_remove,axis=0)
        out_classes = np.delete(out_classes,indices_to_remove,axis=0)
        out_scores = np.delete(out_scores,indices_to_remove,axis=0)        
        
        return out_boxes, out_classes, out_scores

class YOLO(object):
    _defaults = default_config

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        super(YOLO, self).__init__()
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = get_classes(self.classes_path)
        self.anchors = get_anchors(self.anchors_path)
        self.colors = get_colors(self.class_names)
        K.set_learning_phase(0)
        self.inference_model = self._generate_model()

    def _generate_model(self):
        '''to generate the bounding boxes'''
        weights_path = os.path.expanduser(self.weights_path)
        assert weights_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        #YOLOv3 model has 9 anchors and 3 feature layers but
        #Tiny YOLOv3 model has 6 anchors and 2 feature layers,
        #so we can calculate feature layers number to get model type
        num_feature_layers = num_anchors//3

        if num_anchors == 5:
            # YOLOv2 use 5 anchors
            inference_model = get_yolo2_inference_model(self.model_type, self.anchors, num_classes, weights_path=weights_path, input_shape=self.model_image_size + (3,), confidence=0.1)
        else:
            inference_model = get_yolo3_inference_model(self.model_type, self.anchors, num_classes, weights_path=weights_path, input_shape=self.model_image_size + (3,), confidence=0.1)

        inference_model.summary()
        return inference_model

    def predict(self, image_data, image_shape):
        out_boxes, out_scores, out_classes = self.inference_model.predict([image_data, image_shape])

        out_boxes = out_boxes[0]
        out_scores = out_scores[0]
        out_classes = out_classes[0]

        out_boxes = out_boxes.astype(np.int32)
        out_classes = out_classes.astype(np.int32)
        return out_boxes, out_classes, out_scores

    def detect_image(self, image, apply_constraints=False):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'

        image_data = preprocess_image(image, self.model_image_size)

        # prepare origin image shape, (height, width) format
        image_shape = np.array([image.size[1], image.size[0]])
        image_shape = np.expand_dims(image_shape, 0)

        start = time.time()
        out_boxes, out_classes, out_scores = self.predict(image_data, image_shape)
        end = time.time()
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        print("Inference time: {:.8f}s".format(end - start))
        
        if (apply_constraints):
            print('Applying constraints...')
            idx_stem, = np.where(self.class_names == 'stem')
            idx_stick, = np.where(self.class_names == 'stick')
            
            # Stem constraint
            if (len(idx_stick) > 0):            
                stems, = np.where(out_classes == idx_stem)
                
                if (len(stems) > 1):
                    higher_weighted_dist = -9999
                    higher_weighted_dist_idx = -1
                    
                    for i in stems:
                        # Calculating the weighted difference between the stick and the current stem
                        bottom_dist = abs(out_boxes[i][-1] - out_boxes[idx_stick][-1])
                        
                        dist = min([abs(out_boxes[idx_stick][0] - out_boxes[i][1]),
                                    abs(out_boxes[idx_stick][1] - out_boxes[i][0])])
                        
                        weighted_dist = bottom_dist * (1 / (dist + 0.00001)) + bottom_dist * out_scores[i]
                        
                        if (weighted_dist > higher_weighted_dist):
                            higher_weighted_dist = weighted_dist
                            
                            if (higher_weighted_dist_idx >= 0):
                                out_boxes = np.delete(out_boxes,higher_weighted_dist_idx,axis=0)
                                out_classes = np.delete(out_classes,higher_weighted_dist_idx,axis=0)
                                out_scores = np.delete(out_scores,higher_weighted_dist_idx,axis=0)
                            
                            higher_weighted_dist_idx = i

        #draw result on input image
        image_array = np.array(image, dtype='uint8')
        image_array = draw_boxes(image_array, out_boxes, out_classes, out_scores, self.class_names, self.colors)
        return Image.fromarray(image_array)

    def dump_model_file(self, output_model_file):
        self.inference_model.save(output_model_file)

    def dump_saved_model(self, saved_model_path):
        model = self.inference_model
        os.makedirs(saved_model_path, exist_ok=True)

        tf.keras.experimental.export_saved_model(model, saved_model_path)
        print('export inference model to %s' % str(saved_model_path))


#class YOLO_prenms(object):
    #_defaults = default_config

    #@classmethod
    #def get_defaults(cls, n):
        #if n in cls._defaults:
            #return cls._defaults[n]
        #else:
            #return "Unrecognized attribute name '" + n + "'"

    #def __init__(self, **kwargs):
        #super(YOLO_prenms, self).__init__()
        #self.__dict__.update(self._defaults) # set up default values
        #self.__dict__.update(kwargs) # and update with user overrides
        #self.class_names = get_classes(self.classes_path)
        #self.anchors = get_anchors(self.anchors_path)
        #self.colors = get_colors(self.class_names)
        #K.set_learning_phase(0)
        #self.prenms_model = self._generate_model()

    #def _generate_model(self):
        #'''to generate the bounding boxes'''
        #weights_path = os.path.expanduser(self.weights_path)
        #assert weights_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        ## Load model, or construct model and load weights.
        #num_anchors = len(self.anchors)
        #num_classes = len(self.class_names)
        ##YOLOv3 model has 9 anchors and 3 feature layers but
        ##Tiny YOLOv3 model has 6 anchors and 2 feature layers,
        ##so we can calculate feature layers number to get model type
        #num_feature_layers = num_anchors//3

        #prenms_model = get_yolo3_prenms_model(self.model_type, self.anchors, num_classes, weights_path=weights_path, input_shape=self.model_image_size + (3,))

        #return prenms_model


    #def dump_model_file(self, output_model_file):
        #self.prenms_model.save(output_model_file)

    #def dump_saved_model(self, saved_model_path):
        #model = self.prenms_model
        #os.makedirs(saved_model_path, exist_ok=True)

        #tf.keras.experimental.export_saved_model(model, saved_model_path)
        #print('export inference model to %s' % str(saved_model_path))


def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(0 if video_path == '0' else video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    # here we encode the video to MPEG-4 for better compatibility, you can use ffmpeg later
    # to convert it to x264 to reduce file size:
    # ffmpeg -i test.mp4 -vcodec libx264 -f mp4 test_264.mp4
    #
    #video_FourCC    = cv2.VideoWriter_fourcc(*'XVID') if video_path == '0' else int(vid.get(cv2.CAP_PROP_FOURCC))
    video_FourCC    = cv2.VideoWriter_fourcc(*'XVID') if video_path == '0' else cv2.VideoWriter_fourcc(*"mp4v")
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, (5. if video_path == '0' else video_fps), video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image, _ = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release everything if job is finished
    vid.release()
    if isOutput:
        out.release()
    cv2.destroyAllWindows()


def detect_img(yolo,img=None,apply_constraint=False):
    if (img==None):
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image, _ = yolo.detect_image(image,apply_constraints=apply_constraint)
                r_image.show()
    else:
        try:
            image = Image.open(img)
        except:
            raise Exception('Image file does not exisit! Try again!')
        else:
            r_image, boxes = yolo.detect_image(image,apply_constraints=apply_constraint)
            return r_image, boxes


if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='demo or dump out YOLO h5 model')
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_type', type=str,
        help='YOLO model type: yolo3_mobilenet_lite/tiny_yolo3_mobilenet/yolo3_darknet/..., default ' + YOLO.get_defaults("model_type")
    )

    parser.add_argument(
        '--weights_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("weights_path")
    )

    parser.add_argument(
        '--pruning_model', default=False, action="store_true",
        help='Whether to be a pruning model/weights file')

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--model_image_size', type=str,
        help='model image input size as <height>x<width>, default ' +
        str(YOLO.get_defaults("model_image_size")[0])+'x'+str(YOLO.get_defaults("model_image_size")[1]),
        default=str(YOLO.get_defaults("model_image_size")[0])+'x'+str(YOLO.get_defaults("model_image_size")[1])
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )
    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )
    '''
    Command line positional arguments -- for model dump
    '''
    parser.add_argument(
        '--dump_model', default=False, action="store_true",
        help='Dump out training model to inference model'
    )

    parser.add_argument(
        '--output_model_file', type=str,
        help='output inference model file'
    )
    
    parser.add_argument(
        '--input_image', type=str, default="",
        help='[Optional] image file or a folder with multiple images to detect the objects'
    )
    
    parser.add_argument(
        '--apply_constraints', default=False, action="store_true",
        help='[Optional] Apply constraints to the bounding boxes'
    )    

    args = parser.parse_args()
    # param parse
    if args.model_image_size:
        height, width = args.model_image_size.split('x')
        args.model_image_size = (int(height), int(width))
        assert (args.model_image_size[0]%32 == 0 and args.model_image_size[1]%32 == 0), 'model_image_size should be multiples of 32'

    # get wrapped inference object
    yolo = YOLO_np(**vars(args))

    if args.dump_model:
        """
        Dump out training model to inference model
        """
        if not args.output_model_file:
            raise ValueError('output model file is not specified')

        print('Dumping out training model to inference model')
        yolo.dump_model_file(args.output_model_file)
        sys.exit()

    if args.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input_image" in args:
            output_folder = 'example'
            
            dendrometric_valid = dendrometrics.load_dendrometric('utils/dendrometrics_valid.txt')
            dendrometric_list = list()
            
            # Create output folder if it does not exists
            if (not os.path.exists(output_folder)):
                os.makedirs(output_folder)
                
            # Create output folder with the detected crowns
            crown_output_folder = os.path.join(output_folder, 'crown')
            if (not os.path.exists(crown_output_folder)):
                os.makedirs(crown_output_folder)
                
            if (os.path.isfile(args.input_image)):
                file_ = [os.path.basename(args.input_image)]
            else:
                file_ = os.listdir(args.input_image)
            
            for f in file_:
                r_image, boxes = detect_img(yolo,os.path.join(os.path.dirname(args.input_image),f),apply_constraint=args.apply_constraints)
                
                # Calculate dendrometric features                
                if (not dendrometric_valid is None):                    
                    if (args.apply_constraints):
                        idx_ , = np.where(dendrometric_valid[:,0] == f)
                        
                        if (len(idx_) > 0): # If the validation image has dendrometric features
                            tree_height, diameter_crown = dendrometrics.calculate_dendrometrics(boxes[-1], boxes[0], boxes[1])
                            dendrometric_list.append([f, tree_height, diameter_crown, 
                                                      dendrometric_valid[idx_[0],1].astype(float),
                                                      dendrometric_valid[idx_[0],3].astype(float)])
    
                        # Crown segmentation (if it existis)
                        crown = segmentation.crown_segmentation(os.path.join(os.path.dirname(args.input_image),f), 
                                                                boxes[0], 
                                                                boxes[1], 
                                                                yolo.class_names)
                        if (not crown is None):
                            cv2.imwrite(os.path.join(crown_output_folder, f), crown)
                
                out_file = os.path.join(output_folder,f)
                try:
                    r_image.save(out_file,"JPEG",quality=80, optimize=True,progressive=True)
                except:
                    ImageFile.MAXBLOCK = r_image.size[0] * r_image.size[1]
                    r_image.save(out_file, "JPEG", quality=80, optimize=True, progressive=True)
            
            if (len(dendrometric_list) > 0):
                dendrometric_list = np.vstack(dendrometric_list)
                np.savetxt(os.path.join(output_folder, 'dendrometric_features.txt'),dendrometric_list,
                           fmt='%s',delimiter=',', 
                           header='image_file,auto_height,auto_diameter_crown,manual_height,manual_diameter_crown')
        else:    
            if "input" in args:
                print(" Ignoring remaining command line arguments: " + args.input + "," + args.output)
            detect_img(yolo,apply_constraint=args.apply_constraints)
    elif "input" in args:
        detect_video(yolo, args.input, args.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
