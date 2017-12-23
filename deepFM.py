"""
deepFM, using TensorFlow.
"""

from __future__ import division, print_function

import os
import sys
import argparse
import urllib

import numpy as np
import pandas as pd
from pandas.tools.tile import _bins_to_cuts
import pandas.core.algorithms as algos
import tensorflow as tf

#-----------------------------------------------------------------------------

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = {"workclass": 10, "education": 17, "marital_status":8,
                       "occupation": 16, "relationship": 7, "race": 6,
                       "gender": 3, "native_country": 43, "age_binned": 14}
CONTINUOUS_COLUMNS = [ "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]


#-----------------------------------------------------------------------------

class DeepFM(object):
    def __init__(self, max_bins=20, verbose=None, name=None, tensorboard_verbose=3, checkpoints_dir=None):
        '''
        verbose = `bool`
        name = `str` used for run_id (defaults to model_type)
        tensorboard_verbose = `int`: logging level for tensorboard (0, 1, 2, or 3)
        checkpoints_dir = `str`: where checkpoint files will be stored (defaults to "CHECKPOINTS")
        '''
        self.verbose = verbose or 0
        self.tensorboard_verbose = tensorboard_verbose
        self.name = name # name is used for the run_id
        self.data_columns = COLUMNS
        self.continuous_columns = CONTINUOUS_COLUMNS
        self.categorical_columns = CATEGORICAL_COLUMNS	# dict with category_name: category_size
        self.label_column = LABEL_COLUMN
        self.max_bins=max_bins
        self.checkpoints_dir = checkpoints_dir or "CHECKPOINTS"
        if not os.path.exists(self.checkpoints_dir):
            os.mkdir(self.checkpoints_dir)
            print("Created checkpoints directory %s" % self.checkpoints_dir)


    def _qcut(self, x, q, labels=None, retbins=False, precision=3):
        '''
        resolve ValueError: Bin edges must be unique
        '''
        quantiles = np.linspace(0, 1, q + 1)
        bins = algos.quantile(x, quantiles)
        bins = np.unique(bins)
        return _bins_to_cuts(x, bins, labels=labels, retbins=retbins, precision=precision, include_lowest=True)

    def sigmoid(self,logits):
        return (1/(1+tf.exp(-logits)))


    def load_data(self, train_dfn="adult.data", test_dfn="adult.test"):
        '''
        Load data (use files offered in the Tensorflow wide_n_deep_tutorial)
        '''
        if not os.path.exists(train_dfn):
            urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_dfn)
            print("Training data is downloaded to %s" % train_dfn)

        if not os.path.exists(test_dfn):
            urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_dfn)
            print("Test data is downloaded to %s" % test_dfn)

        self.train_data = pd.read_csv(train_dfn, names=COLUMNS, skipinitialspace=True)
        self.test_data = pd.read_csv(test_dfn, names=COLUMNS, skipinitialspace=True, skiprows=1)

        self.train_data[self.label_column] = (self.train_data["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
        self.test_data[self.label_column] = (self.test_data["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

    def prepare_input_data(self, input_data):
        '''
        convert input data to sparse data
        '''

        td=input_data

        #td[self.label_column] = td[self.label_column].astype('category')
        #Y = Y.reshape([-1, 1])
        #Y=pd.get_dummies(td[self.label_column])
        Y=pd.DataFrame(data=td[self.label_column].values, columns=[self.label_column])
        td.drop(self.label_column,axis=1,inplace=True)


        # bin ages (cuts off extreme values)
        age_bins = [ 0, 12, 18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 80, 65535 ]
        td['age_binned'] = pd.cut(td['age'], age_bins, labels=False)
        td = td.replace({'age_binned': {np.nan: 0}})
        td.drop(['age'], axis = 1, inplace = True)
        print ("  %d age bins: age bins = %s" % (len(age_bins), age_bins))

        td=td[self.continuous_columns+self.categorical_columns.keys()]

        for col in self.categorical_columns.keys():
            td[col]=td[col].astype('category')
            td['code_%s' % col]=td[col].cat.codes

        for col in self.continuous_columns:
            if not col in input_data.columns:
                continue
            bins_count = min(self.max_bins, len(td[col].unique()) - 1)
            bins,_=self._qcut(td[col].values, bins_count,labels=False,retbins=True)
            td[("code_%s" % col)]= pd.Series(bins)
            td=td.replace({("code_%s" % col): {np.nan: -1}})

        code_columns=[i for i in td.columns.values if i.startswith('code_')]
        X_codes=td[code_columns].astype(np.int32)
        td.drop(code_columns,axis=1,inplace=True)

        td=pd.get_dummies(td,columns=self.categorical_columns.keys())

        X_mat=td.as_matrix()
        Y_mat=Y.as_matrix()
        X_codes_mat=X_codes.as_matrix()

        return X_mat,Y_mat,X_codes_mat

    def hidden_layer(self,X, name, n_neurons, activation=None):
        with tf.name_scope(name):
            n_input=int(X.get_shape()[1])
            stddev=2/np.sqrt(n_input)
            init=tf.truncated_normal((n_input, n_neurons), stddev=stddev)
            W=tf.Variable(init, name='Weights')
            b=tf.Variable(tf.zeros([n_neurons]), name="Bias")
            z=tf.matmul(X,W)+b
            if activation=='relu':
                return tf.nn.relu(z)
            else:
                return z


    def build_model(self, X,X_code):

        sample_size,feature_size=X.get_shape().as_list()
        feature_code_size=X_code.get_shape().as_list()[1]
        # number of latent factors
        embed_size = 5
        n_hidden1=300
        n_hidden2=100
        n_output=1

        # bias and weights
        w0 = tf.Variable(tf.zeros([1]))
        W = tf.Variable(tf.zeros([feature_size]))

        # interaction factors, randomly initialized
        random_vector = np.random.rand(sample_size, embed_size ).astype(np.float32)
        inter_vector = tf.Variable(random_vector)

        V =tf.reshape(tf.nn.embedding_lookup(inter_vector, X_code),[sample_size,feature_code_size*embed_size])


        second_order = np.zeros([sample_size,1])
        for i in range(feature_code_size):
            for j in range(1, feature_code_size-i):
                former = V[:, i*embed_size:(i+1)*embed_size]
                later = V[:, (i+j)*embed_size:(i+j+1)*embed_size]

                second_order = second_order+tf.reshape(tf.reduce_sum(tf.multiply(former, later), 1, keep_dims=True ),[sample_size,1])


        linear_terms = tf.add(w0,
                    tf.reduce_sum(
                        tf.multiply(W, X), 1, keep_dims=True))

        y_fm = tf.add(linear_terms, second_order)

        with tf.name_scope("dnn"):
            hidden_1=self.hidden_layer(V, "hidden1", n_hidden1, activation="relu")
            hidden_2=self.hidden_layer(hidden_1, "hidden2",n_hidden2, activation="relu")
            logits=self.hidden_layer(hidden_2, "outputs", n_output)

        y_output=tf.add(y_fm,logits)

        return y_output


    def train(self, n_epoch=1000, snapshot_step=10, batch_size=1000,learning_rate=0.01):

        X_mat, Y_mat,X_codes_mat = self.prepare_input_data(self.train_data)
        X_test_mat, Y_test_mat,X_test_codes_mat = self.prepare_input_data(self.test_data)

        sample_size,feature_size=X_mat.shape
        feature_code_size=X_codes_mat.shape[1]
        batch_size = batch_size or Y_mat.shape[0]

        X = tf.placeholder(tf.float32, shape=[batch_size, feature_size])
        X_code = tf.placeholder(tf.int32, [batch_size,feature_code_size])
        Y=tf.placeholder(tf.float32,[batch_size,1])

        y_output=self.build_model(X,X_code)

        cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_output,labels=Y))
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        with tf.name_scope('Accuracy_Calculation'):
            prediction = tf.round(self.sigmoid(y_output)) # calculate accuracy
            predictions_correct = tf.cast(tf.equal(prediction, Y), tf.float32)
            accuracy = tf.reduce_mean(predictions_correct)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            for i in range(n_epoch):
                rand_index = np.random.choice(len(X_mat), size=batch_size)
                rand_x = X_mat[rand_index]
                rand_x_code=X_codes_mat[rand_index]
                rand_y = Y_mat[rand_index]

                _,c=sess.run([optimizer, cost], feed_dict={X: rand_x, X_code:rand_x_code,Y: rand_y})
                train_acc=sess.run(accuracy,feed_dict={X: rand_x, X_code:rand_x_code,Y: rand_y})

                if i % snapshot_step == 0:
                    print("Epoch:", '%04d' % (i+1), "cost=", "{:.9f}".format(c),
                          "Train accuracy=", "{:>6.1%}".format(train_acc))

#-----------------------------------------------------------------------------

def CommandLine(args=None):
    '''
    Main command line.  Accepts args, to allow for simple unit testing.
    '''
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    if args:
        FLAGS.__init__()
        FLAGS.__dict__.update(args)

    try:
        flags.DEFINE_boolean("verbose", True, "Verbose output")
        flags.DEFINE_string("run_name", None, "name for this run (defaults to model type)")
        flags.DEFINE_string("checkpoints_dir", None, "name of directory where checkpoints should be saved")
        flags.DEFINE_integer("n_epoch", 200, "Number of training epoch steps")
    except argparse.ArgumentError:
        pass	# so that CommandLine can be run more than once, for testing

    twad = DeepFM(checkpoints_dir=FLAGS.checkpoints_dir)
    twad.load_data()
    twad.train()
    return twad

#-----------------------------------------------------------------------------

if __name__=="__main__":
    CommandLine()
    None