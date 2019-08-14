import tensorflow as tf
import numpy as np
import math
import reader_writer as rw
import model
import os
import time
import sys
import argparse
from datetime import datetime

#Setting Argument
parser = argparse.ArgumentParser(description='Setting Argument')
parser.add_argument("-m", "--mode", type=str, default='train', required=False)
parser.add_argument("-b", "--batch_size", type=int, default=256, required=False)
parser.add_argument("-e", "--max_epoch", type=int, default=20, required=False)
parser.add_argument("-tm", "--test_mode", type=str, default='multi', required=False)
parser.add_argument("-ts", "--test_step", type=int, default=None, required=False)

#Setting Default Config
args = parser.parse_args()
root_path = './'
data_path = root_path + 'dataset/'
params_dir = 'D:/DeepLearning/params'

classes = 2 # Real or Fake

class Config():

    mode = args.mode

    batch_size = args.batch_size    

    max_epoch = args.max_epoch

    test_mode = args.test_mode

    test_step = args.test_step

    train_step_per_epoch = rw.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
    test_step_per_epoch = rw.NUM_EXAMPLES_PER_EPOCH_FOR_TEST / batch_size

    total_checkpoint = 40

    train_checkpoint = (max_epoch * train_step_per_epoch) / total_checkpoint
    
    test_checkpoint = (max_epoch * test_step_per_epoch) / total_checkpoint

    #trainning rate
    decay = 0.9995
    decay_steps = train_step_per_epoch * 10
    starter_learning_rate = 10e-6


def main():

    config = Config() 

    #rw.create_record(data_path)

    if config.mode == 'train':
        images_train, labels_train = rw.train_inputs(data_dir="./dataset/train.tfrecords", batch_size=config.batch_size,num_epochs=20000)
        
    elif config.mode == 'test':
        images_test, labels_test = rw.test_inputs(data_dir="./dataset/test.tfrecords", batch_size=config.batch_size)
        

    modeler = model.Model(config)

    logits, _ = modeler.inference(classes)

    loss = modeler.loss(logits)

    train_op = modeler.train_op(loss)

    top_k = modeler.cal_accuracy(logits)

    saver = tf.train.Saver(max_to_keep=config.total_checkpoint)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.allocator_type="BFC"

    with tf.Session(config=sess_config) as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
    
        if config.mode == 'train':

            print("Training.....")
        
            for step in range(1,int(config.max_epoch*config.train_step_per_epoch+1)):
                
                start_time = time.time()

                with tf.device("/cpu:0"):
                    image_batch, label_batch = sess.run([images_train, labels_train])

                feed_dict = {
                    modeler.image_holder:image_batch,
                    modeler.label_holder:label_batch,
                    modeler.keep_prob:0.5
                }

                with tf.device("/gpu:0"):
                    _, loss_value = sess.run([train_op, loss],
                                                feed_dict=feed_dict)

                duration = time.time()-start_time

                if step % config.batch_size == 0:
                    examples_per_sec = config.batch_size/duration
                    sec_per_batch = float(duration)

                    format_str = ('step %d, loss =  %.2f (%.1f examples/sec; %.3f sec/batch)')
                    print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
                    print('epoch = ',step / config.train_step_per_epoch)
                
                with tf.device("/gpu:0"):
                    #save checkpoint
                    if step % config.train_checkpoint == 0:
                        saver.save(sess,params_dir+'model',global_step=step)

        elif config.mode == 'test':
            
            print('Testing.....')

            if config.test_mode == None:
                
                num_examples = rw.NUM_EXAMPLES_PER_EPOCH_FOR_TEST
                max_step = int(num_examples/config.batch_size)
                true_count = 0
                total_sample_count = max_step*config.batch_size
                accuracy = np.zeros(classes)
                label_check = np.zeros(classes)
                v_step = 0

                saver.restore(sess,root_path + 'params/model-' + str(config.test_step))

                while v_step < max_step:
                    v_step += 1

                    with tf.device("/cpu:0"):

                        image_batch, label_batch = sess.run([images_test, labels_test])

                        for i in range(config.batch_size):
                            label_check[label_batch[i]] += 1

                    with tf.device("/gpu:0"):
                        predictions = sess.run([top_k], 
                                            feed_dict={modeler.image_holder:image_batch,
                                            modeler.label_holder:label_batch,
                                            modeler.keep_prob:1.0})


                    for i in range(config.batch_size):

                        if predictions[0][i]:

                            accuracy[label_batch[i]] += 1

                    true_count += np.sum(predictions)

                    precision = float(true_count) / total_sample_count

                accuracy = ((accuracy*1.0)/label_check) * 100.0
                print('model-' + str(config.test_step) + '  '  +  str(accuracy),' precision = ',precision)

            elif config.test_mode == 'multi':

                for test_step in range(config.test_step,int(config.test_step*config.total_checkpoint)+1,config.test_step):

                    saver.restore(sess,root_path + 'params/model-' + str(test_step))

                    num_examples = rw.NUM_EXAMPLES_PER_EPOCH_FOR_TEST
                    max_step = int(num_examples/config.batch_size)
                    true_count = 0
                    total_sample_count = max_step*config.batch_size
                    accuracy = np.zeros(classes)
                    label_check = np.zeros(classes)
                    v_step = 0

                    while v_step < max_step:
                        v_step += 1

                        with tf.device("/cpu:0"):

                            image_batch, label_batch = sess.run([images_test, labels_test])

                            for i in range(config.batch_size):
                                label_check[label_batch[i]] += 1

                        with tf.device("/gpu:0"):
                            predictions = sess.run([top_k], 
                                                feed_dict={modeler.image_holder:image_batch,
                                                modeler.label_holder:label_batch,
                                                modeler.keep_prob:1.0})


                        for i in range(config.batch_size):

                            if predictions[0][i]:

                                accuracy[label_batch[i]] += 1

                        true_count += np.sum(predictions)

                        precision = float(true_count) / total_sample_count

                    accuracy = ((accuracy*1.0)/label_check) * 100.0
                    print(' model-' + str(test_step) + '  '  +  str(accuracy),' precision = ',precision)
                    print(label_check)

        coord.request_stop()
        coord.join()

if __name__ == "__main__":
    main()
