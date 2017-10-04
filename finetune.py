from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import math

from inception_v3 import inception_v3, inception_v3_arg_scope
import inception_preprocessing
import hico

slim = tf.contrib.slim

image_size = inception_v3.default_image_size


tf.flags.DEFINE_integer('batch_size', 64, 'Batch size')
tf.flags.DEFINE_integer('epochs', 10, 'Number of training epochs')
tf.flags.DEFINE_float('learning_rate', 1e-2, 'Initial learning rate')

tf.flags.DEFINE_string('log_dir', './logs', 
                        'The directory to save the model files in')
tf.flags.DEFINE_string('dataset_dir', './tfrecords/train',
                        'The directory where the dataset files are stored')
tf.flags.DEFINE_string('checkpoint', './checkpoints',
                        'The directory where the pretrained model is stored')
tf.flags.DEFINE_integer('num_classes', 600,
                        'Number of classes')


FLAGS = tf.app.flags.FLAGS

def get_init_fn(checkpoint_dir):
    checkpoint_exclude_scopes = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(os.path.join(checkpoint_dir, 'inception_v3.ckpt'),
            variables_to_restore)



def main(_):
    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.DEBUG)

        # Select the dataset
        dataset = hico.get_split('train', FLAGS.dataset_dir)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
                        dataset, 
                        num_readers=4,
                        common_queue_capacity=20 * FLAGS.batch_size, 
                        common_queue_min=10 * FLAGS.batch_size)

        image, label = data_provider.get(['image', 'label'])

        label = tf.decode_raw(label, tf.float32)
        
        label = tf.reshape(label, [FLAGS.num_classes])

        # Preprocess images
        image = inception_preprocessing.preprocess_image(image, image_size, image_size,
                is_training=True)

        # Training bathes and queue
        images, labels = tf.train.batch(
                [image, label],
                batch_size = FLAGS.batch_size,
                num_threads = 1,
                capacity = 5 * FLAGS.batch_size)
        
        # Create the model
        with slim.arg_scope(inception_v3_arg_scope()):
            logits, _ = inception_v3(images, num_classes = FLAGS.num_classes, is_training=True)
        
        predictions = tf.nn.sigmoid(logits, name='prediction')
        
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels)
        loss = tf.reduce_mean(cross_entropy)

        # Add summaries
        tf.summary.scalar('loss', loss)

        # Fine-tune only the new layers
        trainable_scopes = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']
        scopes = [scope.strip() for scope in trainable_scopes]
        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        train_op = slim.learning.create_train_op(loss, optimizer, variables_to_train=variables_to_train)

        num_batches = math.ceil(data_provider.num_samples()/float(FLAGS.batch_size)) 
        num_steps = FLAGS.epochs * int(num_batches)
        
        slim.learning.train(
            train_op,
            logdir=FLAGS.log_dir,
            init_fn=get_init_fn(FLAGS.checkpoint),
            number_of_steps=num_steps,
            save_summaries_secs=300,
            save_interval_secs=300
        )

if __name__ == '__main__':
    tf.app.run()
