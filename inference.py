# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2020/7/11 11:31:50
@Author  :   jianghao
@Version :   1.0
@Contact :   haojiang@std.uestc.edu.cn
@Desc    :   Using CycleGAN to generate COVID-19 chest CT image with GGO
'''

import tensorflow as tf
import os
from model import CycleGAN
import utils
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', './data/CANCER2COVID.pb', 'model path (.pb)')
tf.flags.DEFINE_string('input', './test/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860_0000.jpg', 'input image path (.jpg)')
tf.flags.DEFINE_string('output', './reslt/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860_0000.jpg', 'output image path (.jpg)')
tf.flags.DEFINE_integer('image_size', '512', 'image size, default: 256')

def inference(input,output):
  graph = tf.Graph()

  with graph.as_default():
    #with tf.gfile.FastGFile(FLAGS.input, 'rb') as f:
    with tf.gfile.FastGFile(input, 'rb') as f:
      image_data = f.read()
      input_image = tf.image.decode_jpeg(image_data, channels=3)
      input_image = tf.image.resize_images(input_image, size=(FLAGS.image_size, FLAGS.image_size))
      input_image = utils.convert2float(input_image)
      input_image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])

    with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(model_file.read())
    [output_image] = tf.import_graph_def(graph_def,
                          input_map={'input_image': input_image},
                          return_elements=['output_image:0'],
                          name='output')

  with tf.Session(graph=graph) as sess:
    generated = output_image.eval()
    with open(output, 'wb') as f:
    # with open(FLAGS.output, 'wb') as f:
      f.write(generated)

def main(unused_argv):

  path='/home/jianghao/Documents/LUNG/data/0.9'
  output_path='/home/jianghao/Documents/CycleGAN-1/result/0627/'
  filelist = os.listdir(path)
  for files in filelist:
    Olddir = os.path.join(path, files)
    print(Olddir)
    input=Olddir
    output=os.path.join(output_path, files)
    inference(input,output)

if __name__ == '__main__':
  tf.app.run()
