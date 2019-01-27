import tensorflow as tf
import numpy as np
from tfrecord_general import Loader
from dev_tfrecord import DevLoader
import pickle


class Mid_CNN:

    def __init__(self):

        loader = Loader()
        dev_loader = DevLoader()
        iterator_tr, iterator_val = loader.get_dataset()
        iterator_infer = dev_loader.get_dataset()
        iterator_y_bid = self.y_pid_loader()

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
                y_mid = tf.placeholder_with_default(y_mid, shape=(None, 553), name='ymid_placeholder')

                training = tf.placeholder_with_default(True, name='training_bool', shape=())

                flattened_vec = BASE_DNN(x, training)
                input = tf.concat([flattened_vec,y_bid], 1)

                dense1 = tf.layers.dense(inputs=input, units=2106,
                                         kernel_initializer=tf.keras.initializers.he_uniform())
                bn_d1 = tf.contrib.layers.batch_norm(dense1, activation_fn=tf.nn.relu, is_training=training)

                dense_m = tf.layers.dense(inputs=bn_d1, units=553,
                                          kernel_initializer=tf.keras.initializers.he_uniform())

                output_m = tf.nn.softmax(dense_m)

                prediction_m = tf.argmax(output_m, axis=1)

                equality_m = tf.equal(prediction_m, tf.argmax(y_mid, axis=1))

                accuracy_m = tf.reduce_mean(tf.cast(equality_m, tf.float32))

                loss_m = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_mid, logits=dense_m, name='mid_loss'))

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                with tf.control_dependencies(update_ops):
                    step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_m)

            self.loss_m = loss_m
            self.x_placeholder = x
            self.y_b_placeholder = y_bid
            self.y_m_placeholder = y_mid
            self.training = training
            self.accuracy_m = accuracy_m
            self.prediction_m = prediction_m
            self.step = step
            self.iterator_val = iterator_val
            self.iterator_infer = iterator_infer
            self.iterator_y_bid = iterator_y_bid

            tf.summary.scalar("loss", self.loss_m)
            tf.summary.scalar('train accuacy', self.accuracy_m)

        build_base_network()

    def train(self,restore=False):

        saver = tf.train.Saver()
        check_step = 0

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
            try:
                run_id = np.random.randint(0,1e7)
                train_writer = tf.summary.FileWriter(logdir='./logs/'+str(run_id), graph=sess.graph)

                if restore:
                    saver.restore(sess, tf.train.latest_checkpoint('./saves_m'))
                else:
                    sess.run(tf.global_variables_initializer())

                counter = 0

                val_x, val_b, val_m, val_s, val_d = sess.run(self.iterator_val.get_next())

                merge = tf.summary.merge_all()
                print("build cnn model for mid prediction")
                while True:
                    counter += 1
                    _, summary = sess.run([self.step,merge],feed_dict={})
                    train_writer.add_summary(summary,counter)

                    if counter%1000 == 0:
                        acc_m= sess.run([self.accuracy_m],
                                       feed_dict={self.x_placeholder:val_x, self.y_b_placeholder: val_b, self.y_m_placeholder: val_m, self.training:False})
                        print('m :', acc_m[0])
                        accuracy_summary = tf.Summary(value=[tf.Summary.Value(tag='Val_Accuragy',
                                                                              simple_value=acc_m[0])])
                        train_writer.add_summary(accuracy_summary, counter)

                        if acc_m[0] > 0.65:
                            check_step += 1
                            print("saving acc = %f"%acc_m[0],"Saving model ...")
                            save_path = saver.save(sess, './saves_m/model_%d.ckpt'%check_step)

            except KeyboardInterrupt:
                print("Interupted... saving model")

            check_step += 1
            save_path = saver.save(sess, './saves_m/model_%d.ckpt'%check_step)

    def infer(self, batch_size):
        saver = tf.train.Saver()


        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            saver.restore(sess, tf.train.latest_checkpoint('./saves_m'))

            pred_arr = np.array([])
            for i in range((507783 // batch_size) + 1):
                val_x, pid, _, _, _, _ = sess.run(self.iterator_infer.get_next())
                y_bid_feed = sess.run(self.iterator_y_bid.get_next())
                pred = sess.run([self.prediction_m], feed_dict={self.x_placeholder: val_x,
                                                                self.y_b_placeholder: y_bid_feed,
                                                                self.training: False
                                                                })

                print('Tear M, Continuing… {}'.format(i * batch_size))
                pred_arr = np.append(pred_arr, pred)
        return pred_arr

    def load_pkl(self, path):
        with open(path, mode='rb') as in_file:
            return pickle.load(in_file)

    def y_pid_loader(self):
        batch_size = 256

        pred_b = self.load_pkl('infer_b.p')
        y_bid = np.array(pred_b, dtype='int32')

        y_bid_1hot = np.zeros((y_bid.shape[0], 58))
        y_bid_1hot[np.arange(y_bid.shape[0]), y_bid] = 1

        y_bid_data = tf.data.Dataset.from_tensor_slices(y_bid_1hot)
        y_bid_batch = y_bid_data.batch(batch_size)

        return y_bid_batch.make_one_shot_iterator()
