# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert custom dataset to tfrecord with segmentation masks.

Example usage:
    python object_detection/dataset_tools/create_pet_tf_record.py \
        --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""

import sys
import glob
import os

import hashlib
import io
import logging
import os
import random
import re

import contextlib2
from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf
import scipy.io as spio
from random import shuffle

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', 'training/', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'training/egohand_labelmap.pbtxt',
                    'Path to label map proto')
flags.DEFINE_string('mask_type', 'png', 'How to represent instance '
                    'segmentation masks. Options are "png" or "numerical".')

FLAGS = flags.FLAGS


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):

    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def get_class_name_from_filename(file_name):
  """Gets the class name from a file.

  Args:
    file_name: The file name to get the class name from.
               ie. "american_pit_bull_terrier_105.jpg"

  Returns:
    A string of the class name.
  """
  match = re.match(r'([A-Za-z_]+)(_[0-9]+\.jpg)', file_name, re.I)
  return match.groups()[0]

def data_to_tf_example(img_path,
                       mask_path,
                       bbox_path,
                       label_map_dict,
                       mask_type='png'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    img_path: location to rgb image
    mask_path: String path to PNG encoded mask.
    label_map_dict: A map from string label names to integers ids.
    mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to
      smaller file sizes.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """

  #img_path = 'training/images_masks/JENGA_COURTYARD_S_T/frame_0813.jpg'
  #mask_path = 'training/images_masks/JENGA_COURTYARD_S_T/mask_0813.png'
  #bbox_path = 'training/images_masks/JENGA_COURTYARD_S_T/bboxes_0813.mat'
  
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  with tf.gfile.GFile(mask_path, 'rb') as fid:
    encoded_mask_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_mask_png)
  mask = PIL.Image.open(encoded_png_io)
  if mask.format != 'PNG':
    raise ValueError('Mask format not PNG')

  bboxes = loadmat(bbox_path)['bboxes']
  mask_np = np.asarray(mask)

  width = mask.size[0]
  height = mask.size[1]

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []
  masks = []
  
  #mask.show()
  for b in range(bboxes.shape[0]):
      if np.all(bboxes[b]==0):
        continue
      
      xst, yst = bboxes[b][0]-1, bboxes[b][1]-1             # x is along width, y is along height, python 0 based - matlab 1
      yed, xed = yst + bboxes[b][3], xst + bboxes[b][2]     # possible mistake in matlab code

      curr_mask = mask.copy()
      pixel_map = curr_mask.load()
      for r in range(curr_mask.size[0]):    #width
        for c in range(curr_mask.size[1]):    #height
          if r<xst or r>=xed or c<yst or c>=yed:
            pixel_map[r, c] = (0)
      #curr_mask.show()
      output = io.BytesIO()
      curr_mask.save(output, format='PNG')
      masks.append(output.getvalue())
      ymins.append(yst/height)
      xmins.append(xst/width)
      ymaxs.append(yed/height)
      xmaxs.append(xed/width)
      classes.append(1)
      classes_text.append('hand'.encode('utf8'))


  feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(img_path.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(img_path.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/mask' : dataset_util.bytes_list_feature(masks)
  }
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example


def create_tf_record(data_dir, output_filename, label_map_dict, mask_type='png'):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    label_map_dict: The label map dictionary.
    data_dir: Directory where image files are stored.
    mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to
      smaller file sizes.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  dir_names = glob.glob(data_dir+'/*')
  write_cnt = 0
  for dr in dir_names:
    im_files = glob.glob(dr+'/*.jpg')
    shuffle(im_files)
    for im_file in im_files:
      frame_no = os.path.basename(im_file).split('.')[0].split('_')[-1] 
      mask_file = os.path.join(os.path.dirname(im_file), 'mask_'+frame_no+'.png')
      bbox_file = os.path.join(os.path.dirname(im_file), 'bboxes_'+frame_no+'.mat')
      print (im_file, mask_file, bbox_file)
      example = data_to_tf_example(im_file, mask_file, bbox_file, label_map_dict) 
      writer.write(example.SerializeToString())
      print (write_cnt + 1, 'written file', im_file)
      write_cnt += 1
  writer.close()
  '''
  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filename, num_shards)
    for idx, example in enumerate(examples):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples))
      xml_path = os.path.join(annotations_dir, 'xmls', example + '.xml')
      mask_path = os.path.join(annotations_dir, 'trimaps', example + '.png')

      if not os.path.exists(xml_path):
        logging.warning('Could not find %s, ignoring example.', xml_path)
        continue
      with tf.gfile.GFile(xml_path, 'r') as fid:
        xml_str = fid.read()
      xml = etree.fromstring(xml_str)
      data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

      try:
        tf_example = dict_to_tf_example(
            data,
            mask_path,
            label_map_dict,
            image_dir,
            faces_only=faces_only,
            mask_type=mask_type)
        if tf_example:
          shard_idx = idx % num_shards
          output_tfrecords[shard_idx].write(tf_example.SerializeToString())
      except ValueError:
        logging.warning('Invalid example: %s, ignoring.', xml_path)
  '''

# TODO(derekjchow): Add test for pet/PASCAL main files.
def main(_):
  data_dir = FLAGS.data_dir
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  print (label_map_dict)
  logging.info('Reading from hand dataset.')


  train_output_path = os.path.join(FLAGS.output_dir, 'egohand_train_bbox_0idx.record')
  create_tf_record(
      data_dir,
      train_output_path,
      label_map_dict,
      mask_type=FLAGS.mask_type)


if __name__ == '__main__':
  tf.app.run()
