import tensorflow as tf
import numpy as np

print(tf.__version__)

from tensorflow.contrib.learn.python.learn.datasets import base

# declare data files
IRIS_TRAINING_DATA = "iris_training.csv"
IRIS_TEST_DATA = "iris_test.csv"

# load datasets from csv
training_set = base.load_csv_with_header(filename=IRIS_TRAINING_DATA, features_dtype=np.float32, target_dtype=np.int)
test_set = base.load_csv_with_header(filename=IRIS_TEST_DATA, features_dtype=np.float32, target_dtype=np.int)

print(training_set.data)

print(training_set.target)

# Specify that all features have real-value data
feature_name = "flower_features"
feature_columns = [tf.feature_column.numeric_column(feature_name, shape=[4])]

classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=3, model_dir="/tmp/iris_model")

def input_fn(dataset):
    def _fn():
        features = {feature_name: tf.constant(dataset.data)}
        label = tf.constant(dataset.target)
        return features, label
    return _fn

print(input_fn(training_set)())

# fit model
classifier.train(input_fn=input_fn(training_set), steps=1000)
print("fit done")

# evaluate accuracy
accuracy_score = classifier.evaluate(input_fn=input_fn(test_set), steps=100)["accuracy"]
print("\nAccuracy: {0:f}".format(accuracy_score))

# export model
feature_spec = {'flower_features': tf.FixedLenFeature(shape=[4], dtype=np.float32)}

serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

classifier.export_savedmodel(export_dir_base='/tmp/iris_model' + '/export',
                            serving_input_receiver_fn=serving_fn)
print("\nExport done")
