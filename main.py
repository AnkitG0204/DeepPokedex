import os
import json
import argparse
import pickle

from pyspark import SparkContext
from zoo import create_spark_conf
from zoo import init_engine

from zoo.common.utils import Sample
from zoo.pipeline.api import autograd
from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.keras.layers import Input
from zoo.pipeline.api.keras.layers import Dense
from zoo.pipeline.api.keras.layers import Flatten
from zoo.pipeline.api.keras.layers import Dropout
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import TimeDistributed
from zoo.pipeline.api.keras.layers import Convolution2D
from zoo.pipeline.api.keras.layers import AveragePooling2D
from zoo.pipeline.api.keras.layers import L2Regularizer
from zoo.pipeline.api.keras.layers import BatchNormalization

from bigdl.nn.criterion import CrossEntropyCriterion
from bigdl.optim.optimizer import Optimizer
from bigdl.optim.optimizer import Adam
from bigdl.optim.optimizer import MaxEpoch
from bigdl.optim.optimizer import EveryEpoch
from bigdl.optim.optimizer import Top1Accuracy
from bigdl.optim.optimizer import TrainSummary
from bigdl.optim.optimizer import ValidationSummary

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", "--dd", type=str, default="./data")
parser.add_argument("--save_dir", "--sd", type=str, default="./save")
parser.add_argument("--num_epoch", "-ne", type=int, default=64)
parser.add_argument("--batch_size", "-bs", type=int, default=32)
parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
parser.add_argument("--penalty_rate", "-pr", type=float, default=1e-6)
parser.add_argument("--dropout_rate", "-dr", type=float, default=0.75)

LAYER_1_NUM_CHANNEL = 8         # Convolutional Channels in 1st Layer.
CONVOLVE_1_KERNEL_SIZE = 9      # Window size of the first layer of convolution kernel.
POOLING_1_WINDOW_SIZE = 2       # The window size of the first pooling layer.
POOLING_1_STRIDE_SIZE = 2       # Sliding step size of the first pooling layer.
LAYER_2_NUM_CHANNEL = 2         # Convolution channels in 2nd layer.
CONVOLVE_2_KERNEL_SIZE = 5      # Window size of the second layer of convolution kernel.
POOLING_2_WINDOW_SIZE = 2       # The window size of the second pooling layer.
POOLING_2_STRIDE_SIZE = 2       # Sliding step size of the second pooling layer.
FC_LINEAR_DIMENSION = 64        # Dimension of the fully connected layer.

args = parser.parse_args()
print (json.dumps(args.__dict__, indent=True, ensure_ascii=False))

# Pyspark + Analytic-Zoo.
sc = SparkContext.getOrCreate(
    conf=create_spark_conf()
    .setMaster("local[16]")
    .set("spark.driver.memory", "512m")
    .setAppName("OneShotLearning")
)
init_engine()

# Use the pickle file as input features and labels.
train_img = pickle.load(
    open(os.path.join(args.data_dir.encode('utf-8').strip(), "train_image.pkl"), "r")
)
train_lbl = pickle.load(
    open(os.path.join(args.data_dir.encode('utf-8').strip(), "train_label.pkl"), "r")
)
test_img = pickle.load(
    open(os.path.join(args.data_dir.encode('utf-8').strip(), "test_image.pkl"), "r")
)
test_lbl = pickle.load(
    open(os.path.join(args.data_dir.encode('utf-8').strip(), "test_label.pkl"), "r")
)

# Modelling Starts. Defining required variables.
t_train_img = train_img.transpose((0, 1, 4, 2, 3)) / 225.0
t_test_img = test_img.transpose((0, 1, 4, 2, 3)) / 225.0

NUM_TRAIN_SMP, _, IMAGE_SIZE, _, NUM_IMAGE_CHANNEL = train_img.shape
NUM_TEST_SMP, NUM_CLASS_LABEL, _, _, _ = test_img.shape

# Making the RDD. (Resilient Distributed Datasets - DS for Apache Spark)
train_rdd = sc.parallelize(t_train_img).zip(sc.parallelize(train_lbl)).map(
    lambda featurelabel: Sample.from_ndarray(featurelabel[0], featurelabel[1] + 1)
)
test_rdd = sc.parallelize(t_test_img).zip(sc.parallelize(test_lbl)).map(
    lambda featurelabel: Sample.from_ndarray(featurelabel[0], featurelabel[1] + 1)
)

# Making a Zoo-Keras Pipeline with a CNN model.
input_shape = (NUM_CLASS_LABEL, NUM_IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
both_input = Input(shape=input_shape)

convolve_net = Sequential()
convolve_net.add(Convolution2D(
    nb_filter=LAYER_1_NUM_CHANNEL,      # 4 -> 8.
    nb_row=CONVOLVE_1_KERNEL_SIZE,      # Size: 32 - 9 + 1 = 24
    nb_col=CONVOLVE_1_KERNEL_SIZE,
    activation="relu",
    input_shape=(
        NUM_IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE
    ),
    W_regularizer=L2Regularizer(
        args.penalty_rate
    )
))
convolve_net.add(AveragePooling2D(
    pool_size=(
        POOLING_1_WINDOW_SIZE,          # Size: 24 / 2 = 12.
        POOLING_1_WINDOW_SIZE
    ),
    strides=(
        POOLING_1_STRIDE_SIZE,
        POOLING_1_STRIDE_SIZE
    )
))
convolve_net.add(BatchNormalization())
convolve_net.add(Convolution2D(
    nb_filter=LAYER_2_NUM_CHANNEL,      # 8 -> 2.
    nb_row=CONVOLVE_2_KERNEL_SIZE,      # Size: 12 - 5 + 1 = 8.
    nb_col=CONVOLVE_2_KERNEL_SIZE,
    activation="relu",
    W_regularizer=L2Regularizer(
        args.penalty_rate
    )
))
convolve_net.add(AveragePooling2D(
    pool_size=(
        POOLING_2_WINDOW_SIZE,          # Size: 8 / 2 = 4.
        POOLING_2_WINDOW_SIZE
    ),
    strides=(
        POOLING_2_STRIDE_SIZE,
        POOLING_2_STRIDE_SIZE
    ),
))
convolve_net.add(BatchNormalization())
convolve_net.add(Flatten())             # Size: 4 * 4 * 2 -> 32
convolve_net.add(Dense(
    output_dim=FC_LINEAR_DIMENSION,     # Size: 32 -> 64.
    activation="sigmoid",
    W_regularizer=L2Regularizer(
        args.penalty_rate
    )
))
convolve_net.add(Dropout(args.dropout_rate))



'''

To be implemented in Review - 3.

- Parameter sharing by Intel BigDL to further increase the efficiency of Distributed Processing.
- To implement the ‘Difference’ function of Siamese networks.
- Optimisation of the Distributed CNN model.
- Minor Bug Fixes
- Logging(Optional)

'''
