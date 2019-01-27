import tensorflow as tf
import numpy as np
#from hdf5_to_tfrecord import Loader
from tfrecord_general import Loader
from dev_tfrecord import DevLoader


class Bid_CNN:
    def __init__(self):

        loader = Loader()
        dev_loader = DevLoader()
        iterator_tr, iterator_val = loader.get_dataset()
        iterator_infer = dev_loader.get_dataset()

        def BASE_DNN(x, training):
            # conv1d - 2 layers
            W1 = tf.get_variable('W1', [5, 2048, 1024],
                                initializer=tf.contrib.layers.variance_scaling_initializer())
            conv1 = tf.nn.conv1d(x, W1, stride=2, padding='SAME')
            bnc1_1 = tf.contrib.layers.batch_norm(conv1, activation_fn=tf.nn.relu, is_training=training)
            conv1 = tf.nn.relu(bnc1_1)

            W2 = tf.get_variable('W2', [5, 1024, 512],
                                 initializer=tf.contrib.layers.variance_scaling_initializer())
            conv2 = tf.nn.conv1d(conv1, W2, stride=2, padding='SAME')
            bnc1_2 = tf.contrib.layers.batch_norm(conv2, activation_fn=tf.nn.relu, is_training=training)
            conv2 = tf.reshape(bnc1_2, [-1, 512])
            conv2 = tf.nn.relu(conv2)

            #DNN - 2 layers
            dense_l1 = tf.layers.dense(inputs=conv2, units=2048,
                                     kernel_initializer=tf.keras.initializers.he_uniform())
            bn1_1 = tf.contrib.layers.batch_norm(dense_l1, activation_fn=tf.nn.relu, is_training=training)
            dense_l2 = tf.layers.dense(inputs=bn1_1, units=2048,
                                       kernel_initializer=tf.keras.initializers.he_uniform())
            bn1_2 = tf.contrib.layers.batch_norm(dense_l2, activation_fn=tf.nn.relu, is_training=training)

            dropout = tf.layers.dropout(bn1_2, training=training, rate=0.7)

            flattened = tf.contrib.layers.flatten(dropout)
            return flattened

        def build_base_network():
            with tf.device('/device:GPU:0'):

                x_data, y_bid, y_mid, y_sid, y_did = iterator_tr.get_next()

                x = tf.placeholder_with_default(x_data, shape=(None, 1, 2048), name='x_placeholder')
                y_bid = tf.placeholder_with_default(y_bid, shape=(None, 58), name='ybid_placeholder')

                training = tf.placeholder_with_default(True, name='training_bool', shape=())

                flattened_vec = BASE_DNN(x, training)

                dense1 = tf.layers.dense(inputs=flattened_vec, units=2048,
                                         kernel_initializer=tf.keras.initializers.he_uniform())
                bn_d1 = tf.contrib.layers.batch_norm(dense1, activation_fn=tf.nn.relu, is_training=training)

                dense_b = tf.layers.dense(inputs=bn_d1, units=58,
                                          kernel_initializer=tf.keras.initializers.he_uniform())

                output_b = tf.nn.softmax(dense_b)

                prediction_b = tf.argmax(output_b, axis=1)

                equality_b = tf.equal(prediction_b, tf.argmax(y_bid, axis=1))

                accuracy_b = tf.reduce_mean(tf.cast(equality_b, tf.float32))

                loss_b = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_bid, logits=dense_b, name='bid_loss'))

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                with tf.control_dependencies(update_ops):
                    step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_b)

            self.loss_b = loss_b
            self.x_placeholder = x
            self.y_b_placeholder = y_bid
            self.y_m_placeholder = y_mid
            self.y_s_placeholder = y_sid
            self.y_d_placeholder = y_did
            self.training = training
            self.accuracy_b = accuracy_b
            self.prediction_b = prediction_b
            self.step = step
            self.iterator_val = iterator_val
            self.iterator_infer = iterator_infer

            tf.summary.scalar("loss", self.loss_b)
            tf.summary.scalar('train accuacy', self.accuracy_b)

        build_base_network()

    def train(self,restore=False):

        saver = tf.train.Saver()
        check_step = 0

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
            try:
                run_id = np.random.randint(0,1e7)
                train_writer = tf.summary.FileWriter(logdir='./logs/'+str(run_id), graph=sess.graph)

                if restore:
                    saver.restore(sess, tf.train.latest_checkpoint('./saves_t1'))
                else:
                    sess.run(tf.global_variables_initializer())

                counter = 0

                val_x, val_b, val_m, val_s, val_d = sess.run(self.iterator_val.get_next())

                merge = tf.summary.merge_all()
                print("build cnn model for bid prediction")
                while True:
                    counter += 1
                    _, summary = sess.run([self.step,merge],feed_dict={})
                    train_writer.add_summary(summary,counter)

                    if counter%1000 == 0:
                        acc_b= sess.run([self.accuracy_b],
                                       feed_dict={self.x_placeholder:val_x, self.y_b_placeholder: val_b, self.y_m_placeholder: val_m,
                                                  self.y_s_placeholder: val_s, self.y_d_placeholder: val_d, self.training:False})
                        print('b :', acc_b[0])
                        accuracy_summary = tf.Summary(value=[tf.Summary.Value(tag='Val_Accuragy',
                                                                              simple_value=acc_b[0])])
                        train_writer.add_summary(accuracy_summary, counter)

                        if acc_b[0] > 0.81:
                            check_step += 1
                            print("saving acc = %f"%acc_b[0],"Saving model ...")
                            save_path = saver.save(sess, './saves_t1/model_%d.ckpt'%check_step)

            except KeyboardInterrupt:
                print("Interupted... saving model")

            check_step += 1
            save_path = saver.save(sess, './saves_t1/model_%d.ckpt'%check_step)

    def infer(self, batch_size):
        saver = tf.train.Saver()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
            saver.restore(sess, tf.train.latest_checkpoint('./saves_t1'))

            pid_arr = np.array([])
            pred_arr = np.array([])
            for i in range((507783//batch_size) + 1):
                val_x, pid, val_b, val_m, val_s, val_d = sess.run(self.iterator_infer.get_next())
                pred = sess.run([self.prediction_b], feed_dict={self.x_placeholder: val_x, self.training: False})

                print('Tear B, Continuing... {}'.format(i*batch_size))
                pred_arr = np.append(pred_arr, pred)
                pid_arr = np.append(pid_arr, pid)

        return pred_arr, pid_arr


