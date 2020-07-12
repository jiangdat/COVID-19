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
from tensorflow.python.tools.freeze_graph import freeze_graph
from model import CycleGAN
import utils
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint_dir', './checkpoints/20200607-2330', 'checkpoints directory path')
tf.flags.DEFINE_string('XtoY_model', './data/COVID2CANCER.pb', 'XtoY model name, default: ./data/COVID2CANCER.pb')
tf.flags.DEFINE_string('YtoX_model', './data/CANCER2COVID.pb', 'YtoX model name, default: ./data/CANCER2COVID.pb')
tf.flags.DEFINE_integer('image_size', '512', 'image size, default: 512')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')

def export_graph(model_name, XtoY=True):
  graph = tf.Graph()

  with graph.as_default():
    cycle_gan = CycleGAN(ngf=FLAGS.ngf, norm=FLAGS.norm, image_size=FLAGS.image_size)

    input_image = tf.placeholder(tf.float32, shape=[FLAGS.image_size, FLAGS.image_size, 3], name='input_image')
    cycle_gan.model()
    if XtoY:
      output_image = cycle_gan.G.sample(tf.expand_dims(input_image, 0))
    else:
      output_image = cycle_gan.F.sample(tf.expand_dims(input_image, 0))

    output_image = tf.identity(output_image, name='output_image')
    restore_saver = tf.train.Saver()
    export_saver = tf.train.Saver()

  with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    restore_saver.restore(sess, latest_ckpt)
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [output_image.op.name])

    tf.train.write_graph(output_graph_def, 'pretrained', model_name, as_text=False)

def main(unused_argv):
  print('Export XtoY model...')
  export_graph(FLAGS.XtoY_model, XtoY=True)
  print('Export YtoX model...')
  export_graph(FLAGS.YtoX_model, XtoY=False)

if __name__ == '__main__':
  tf.app.run()
