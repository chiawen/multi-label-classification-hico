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


tf.flags.DEFINE_integer('batch_size', 256, 'Batch size')

tf.flags.DEFINE_string('dataset_dir', './tfrecords/test',
                        'The directory where the dataset files are stored')
tf.flags.DEFINE_string('checkpoint', './logs',
                        'The directory where the pretrained model is stored')
tf.flags.DEFINE_integer('num_classes', 600,
                        'Number of classes')


FLAGS = tf.app.flags.FLAGS

# reference: https://github.com/broadinstitute/keras-rcnn/issues/6
def calculate_mAP(y_pred, y_true):
    num_classes = y_true.shape[1]
    average_precisions = []

    for index in range(FLAGS.num_classes):
        pred = y_pred[:,index]
        label = y_true[:,index]

        """
        positive_indices = pred > 0.5
        pred = pred[positive_indices]
        label = label[positive_indices]
        print(pred.shpae)
        """

        sorted_indices = np.argsort(-pred)
        sorted_pred = pred[sorted_indices]
        sorted_label = label[sorted_indices]

        

        tp = (sorted_label == 1)
        fp = (sorted_label == 0)

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        npos = np.sum(sorted_label)
        
        recall = tp * 1.0 / npos

        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        precision = tp*1.0 / np.maximum((tp + fp), np.finfo(np.float64).eps)

        mrec = np.concatenate(([0.], recall, [1.])) 
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        average_precisions.append(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))

    print(average_precisions)
    mAP = np.mean(average_precisions)

    return mAP

def main(_):
    
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        # Select the dataset
        dataset = hico.get_split('test', FLAGS.dataset_dir)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
                        dataset, 
                        num_readers=1,
                        common_queue_capacity=20 * FLAGS.batch_size, 
                        common_queue_min=10 * FLAGS.batch_size,
                        shuffle=False)

        image, label = data_provider.get(['image', 'label'])
        
        label = tf.decode_raw(label, tf.float32)
        
        label = tf.reshape(label, [FLAGS.num_classes])

        
        # Preprocess images
        image = inception_preprocessing.preprocess_image(image, image_size, image_size,
                is_training=False)

        # Training bathes and queue
        images, labels = tf.train.batch(
                [image, label],
                batch_size = FLAGS.batch_size,
                num_threads = 1,
                capacity = 5 * FLAGS.batch_size,
                allow_smaller_final_batch=True)
        
       
        # Create the model
        with slim.arg_scope(inception_v3_arg_scope()):
            logits, _ = inception_v3(images, num_classes = FLAGS.num_classes, is_training=False)
        
        predictions = tf.nn.sigmoid(logits, name='prediction')
        
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels)
        loss = tf.reduce_mean(cross_entropy)

        correct_prediction = tf.equal(tf.round(predictions), labels)
        
        # Mean accuracy over all labels:
        # http://stackoverflow.com/questions/37746670/tensorflow-multi-label-accuracy-calculation
        accuracy = tf.cast(correct_prediction, tf.float32)
        mean_accuracy = tf.reduce_mean(accuracy)


        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint)
        init_fn = slim.assign_from_checkpoint_fn(
            checkpoint_path,
            slim.get_variables_to_restore())

        num_batches = math.ceil(data_provider.num_samples()/float(FLAGS.batch_size))

        prediction_list = []
        label_list = []
        count = 0
        
        with tf.Session() as sess:
            with slim.queues.QueueRunners(sess):
                sess.run(tf.local_variables_initializer())
                init_fn(sess)
                
                for step in range(int(num_batches)):
                    np_loss, np_accuracy, np_logits, np_prediction, np_labels = sess.run(
                            [loss, mean_accuracy, logits, predictions, labels]) 
                    
                    prediction_list.append(np_prediction)
                    label_list.append(np_labels)
                    
                    count += np_labels.shape[0]
                    
                    print('Step {}, count {}, loss: {}'.format(step,count,  np_loss))
                    
        
        prediction_arr = np.concatenate(prediction_list, axis=0)
        label_arr = np.concatenate(label_list, axis=0)
        print(prediction_arr.shape)
        print(label_arr.shape)
        
        mAP = calculate_mAP(prediction_arr, label_arr)
        print('mAP score: {}'.format(mAP))

if __name__ == '__main__':
    tf.app.run()
