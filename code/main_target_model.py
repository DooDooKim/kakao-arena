from t1_model_cnn import *
from t2_model_cnn import *
from t3_model_cnn import *
from t4_model_cnn import *

# from Middle_model import *
import numpy as np
import pickle


def load_pkl(path):
    with open(path, mode='rb') as in_file:
        return pickle.load(in_file)


def main():
    # target_dnn = Bid_CNN()
    #target_dnn = Mid_CNN()
    #target_dnn = Sid_CNN()
    #target_dnn = Did_CNN()
    #target_dnn = Mid_CNN()

    #target_dnn = Bayesian_CNN()
    #
    # target_dnn.train(restore=True)
    # target_dnn.infer()

    # tb = Bid_CNN()
    # pred_b, pid = tb.infer(256)
    # print(np.shape(pid))
    # print(np.shape(pred_b))
    # pickle.dump(pid, open('./infer_pid.p', 'wb'))
    # pickle.dump(pred_b, open('./infer_b.p', 'wb'))

    tm = Mid_CNN()
    pred_m = tm.infer(256)
    print(np.shape(pred_m))
    pickle.dump(pred_m, open('./infer_m.p', 'wb'))

    # ts = Sid_CNN()
    # pred_s = ts.infer(256)
    # print(np.shape(pred_s))
    # pickle.dump(pred_s, open('./infer_s.p', 'wb'))
    #
    # td = Did_CNN()
    # pred_d = td.infer(256)
    # print(np.shape(pred_d))
    # pickle.dump(pred_d, open('./infer_d.p', 'wb'))

    # #중분류 용진모델
    # tear_m = Mid_CNN()
    # pred_m = tear_m.infer(256)
    # print(np.shape(pred_m))
    # pickle.dump(pred_m, open('./infer_m_new.p', 'wb'))


if __name__ == "__main__":
    main()
