import tensorflow as tf
import h5py
import os.path

def read_hdf5(file):
    hlv = h5py.File(file, "r")
    hdf5 = hlv['train']
    return hdf5

class Loader():
    def __init__(self, batch_size=256):
        self.filenames =["./data/kakao01.tfrecords", "./data/kakao02.tfrecords", "./data/kakao03.tfrecords",
                         "./data/kakao04.tfrecords", "./data/kakao05.tfrecords", "./data/kakao06.tfrecords",
                         "./data/kakao07.tfrecords", "./data/kakao08.tfrecords", "./data/kakao09.tfrecords"]
        self.cnn_dec = True
        found = True
        v_count = 0
        for file in self.filenames:
            if not os.path.isfile(file):
                found = False
                v_count+=1
                print('no kakao.tfrecord file')

        if not found:
            v_count = 9 - v_count
            for idx_n in range(v_count, 9):
                hdf5_path = 'C:/Users/user/Desktop/kakao/data/train.chunk.0%d'%(idx_n+1)
                batch_tr_data = read_hdf5(hdf5_path)
                img_feat = batch_tr_data[b'img_feat']
                bcateid = batch_tr_data[b'bcateid']
                mcateid = batch_tr_data[b'mcateid']
                scateid = batch_tr_data[b'scateid']
                dcateid = batch_tr_data[b'dcateid']
                print('\n','kakao %d file is writing'%(idx_n+1))
                self.create_tf_record(img_feat=img_feat,
                                      bcateid=bcateid,
                                      mcateid=mcateid,
                                      scateid=scateid,
                                      dcateid=dcateid,
                                      path='kakao0%d.tfrecords'%(idx_n+1))
                print('done!','\n')
        self.batch_size = batch_size

    def create_tf_record(self, img_feat, bcateid, mcateid, scateid, dcateid, path):

        with tf.python_io.TFRecordWriter(path) as writer:
            for i in range(img_feat.shape[0]):
                if i % 2000 == 0: print(i+1,' / ',str(img_feat.shape[0]))
                img_f = img_feat[i].tostring()
                bid = bcateid[i]
                mid = mcateid[i]
                if scateid[i] == -1: sid = 0
                else: sid = scateid[i]
                if dcateid[i] == -1: did = 0
                else: did = dcateid[i]

                features = tf.train.Features(
                    feature = {
                        'img_feat': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_f])),
                        'bcateid': tf.train.Feature(int64_list=tf.train.Int64List(value=[bid])),
                        'mcateid': tf.train.Feature(int64_list=tf.train.Int64List(value=[mid])),
                        'scateid': tf.train.Feature(int64_list=tf.train.Int64List(value=[sid])),
                        'dcateid': tf.train.Feature(int64_list=tf.train.Int64List(value=[did]))
                    }
                )
                batch_data = tf.train.Example(features=features)
                serialized = batch_data.SerializeToString()
                writer.write(serialized)

    def get_dataset(self, train=True, cnn=True):
            if train:
                filenames_tr = self.filenames[0:7]
                filenames_val = self.filenames[7:10]
            if cnn:
                self.cnn_dec=True
            else:
                self.cnn_dec=False
            dataset_tr = tf.data.TFRecordDataset(filenames_tr)
            dataset_val = tf.data.TFRecordDataset(filenames_val)

            dataset_tr = dataset_tr.apply(tf.contrib.data.shuffle_and_repeat(10000,seed=0))
            dataset_tr = dataset_tr.apply(tf.contrib.data.map_and_batch(self.parse_example, batch_size=self.batch_size, num_parallel_batches=9))
            dataset_tr = dataset_tr.apply(tf.contrib.data.prefetch_to_device('/device:GPU:0',100))
            iterator_tr = dataset_tr.make_one_shot_iterator()

            dataset_val = dataset_val.apply(tf.contrib.data.shuffle_and_repeat(10000, seed=0))
            dataset_val = dataset_val.apply(tf.contrib.data.map_and_batch(self.parse_example, batch_size=self.batch_size, num_parallel_batches=9))
            #dataset_val = dataset_val.apply(tf.contrib.data.prefetch_to_device('./device:GPU:0',100))
            iterator_val = dataset_val.make_one_shot_iterator()

            return iterator_tr, iterator_val

    def parse_example(self, serialized):

        features = tf.parse_single_example(serialized=serialized, features={
            'img_feat': (tf.FixedLenFeature((), tf.string, default_value="")),
            'bcateid': (tf.FixedLenFeature((), tf.int64, default_value=0)),
            'mcateid': (tf.FixedLenFeature((), tf.int64, default_value=0)),
            'scateid': (tf.FixedLenFeature((), tf.int64, default_value=0)),
            'dcateid': (tf.FixedLenFeature((), tf.int64, default_value=0))})

        raw_image = features['img_feat']
        image = tf.decode_raw(raw_image, tf.float32)

        if self.cnn_dec:
            return tf.reshape(image,[1,2048]), tf.one_hot(features['bcateid'],58), \
                   tf.one_hot(features['mcateid'],553), tf.one_hot(features['scateid'],3191), tf.one_hot(features['dcateid'],405)
        else:
            return image, tf.one_hot(features['bcateid'],58), tf.one_hot(features['mcateid'],553), \
                   tf.one_hot(features['scateid'],3191), tf.one_hot(features['dcateid'],405)

