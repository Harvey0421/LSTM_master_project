# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf
import numpy as np
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print(tf.test.is_built_with_cuda())
    data = np.load('../data/pretrain/NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy')
    print(data)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
