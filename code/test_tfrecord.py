import tensorflow as tf
import h5py
import os.path

def read_hdf5(file):
    hlv = h5py.File(file, "r")
    hdf5 = hlv['test']
    return hdf5

class TestLoader():
    def __init__(self, batch_size=256):
        self.filenames =["./data/test01.tfrecords", "./data/test02.tfrecords"]
        self.cnn_dec = True
        found = True
        for file in self.filenames:
            if not os.path.isfile(file):
                found = False
                print('no test.tfrecord file')

        if not found:
            for i in range (2):
                hdf5_path = 'C:/Users/user/Desktop/kakao/data/test.chunk.0%d'%(i+1)
                batch_tr_data = read_hdf5(hdf5_path)
                img_feat = batch_tr_data[b'img_feat']
                pid = batch_tr_data[b'pid']
                bcateid = batch_tr_data[b'bcateid']
                mcateid = batch_tr_data[b'mcateid']
                scateid = batch_tr_data[b'scateid']
                dcateid = batch_tr_data[b'dcateid']
                print('\n','test%d file is writing'%(i+1))
                self.create_tf_record(img_feat=img_feat,
                                      pid=pid,
                                      bcateid=bcateid,
                                      mcateid=mcateid,
                                      scateid=scateid,
                                      dcateid=dcateid,
                                      path='test0%d.tfrecords'%(i+1))
                print('done!','\n')
        self.batch_size = batch_size

    def create_tf_record(self, img_feat, pid, bcateid, mcateid, scateid, dcateid, path):

        with tf.python_io.TFRecordWriter(path) as writer:
            for i in range(img_feat.shape[0]):
                if i % 2000 == 0: print(i+1,' / ',str(img_feat.shape[0]))
                img_f = img_feat[i].tostring()
                pid_p = pid[i].tostring()
                bid = bcateid[i]
                mid = mcateid[i]
                if scateid[i] == -1: sid = 0
                else: sid = scateid[i]
                if dcateid[i] == -1: did = 0
                else: did = dcateid[i]

                features = tf.train.Features(
                    feature = {
                        'img_feat': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_f])),
                        'pid': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pid_p])),
                        'bcateid': tf.train.Feature(int64_list=tf.train.Int64List(value=[bid])),
                        'mcateid': tf.train.Feature(int64_list=tf.train.Int64List(value=[mid])),
                        'scateid': tf.train.Feature(int64_list=tf.train.Int64List(value=[sid])),
                        'dcateid': tf.train.Feature(int64_list=tf.train.Int64List(value=[did]))
                    }
                )
                batch_data = tf.train.Example(features=features)
                serialized = batch_data.SerializeToString()
                writer.write(serialized)

    def get_dataset(self, cnn=True):
            filenames_te = self.filenames
            if cnn:
                self.cnn_dec=True
            else:
                self.cnn_dec=False
            dataset_te = tf.data.TFRecordDataset(filenames_te)

            dataset_te = dataset_te.apply(tf.contrib.data.map_and_batch(self.parse_example, batch_size=self.batch_size, num_parallel_batches=3))
            iterator_te = dataset_te.make_one_shot_iterator()

            return iterator_te

    def parse_example(self, serialized):

        features = tf.parse_single_example(serialized=serialized, features={
            'img_feat': (tf.FixedLenFeature((), tf.string, default_value="")),
            'pid': (tf.FixedLenFeature((), tf.string, default_value="")),
            'bcateid': (tf.FixedLenFeature((), tf.int64, default_value=0)),
            'mcateid': (tf.FixedLenFeature((), tf.int64, default_value=0)),
            'scateid': (tf.FixedLenFeature((), tf.int64, default_value=0)),
            'dcateid': (tf.FixedLenFeature((), tf.int64, default_value=0))})

        raw_image = features['img_feat']
        image = tf.decode_raw(raw_image, tf.float32)

        #pid = tf.decode_raw(features['pid'], tf.string)

        if self.cnn_dec:
            return tf.reshape(image,[1,2048]),features['pid'], tf.one_hot(features['bcateid'],58), \
                   tf.one_hot(features['mcateid'],553), tf.one_hot(features['scateid'],3191), tf.one_hot(features['dcateid'],405)
        else:
            return image, features['pid'], tf.one_hot(features['bcateid'],58), tf.one_hot(features['mcateid'],553), \
                   tf.one_hot(features['scateid'],3191), tf.one_hot(features['dcateid'],405)

TestLoader()