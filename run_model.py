"""
Small script to run the regression model as a standalone code for training and testing purposes
"""
import ConfigParser
import os
import numpy
from tf_regression import TensorFlowRegressionModel

# get config file
HERE = os.path.dirname(os.path.realpath(__file__))
Config = ConfigParser.ConfigParser()
Config.read(HERE + '/settings.ini')
# settings for the training
MODEL_DIR = Config.get('model', 'LOCAL_MODEL_FOLDER')
LEARNING_RATE = float(Config.get('model', 'LEARNING_RATE'))
TRAINING_EPOCHS = int(Config.get('model', 'TRAINING_EPOCHS'))


def main():
    # training data
    train_X = numpy.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                             7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_Y = numpy.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                             2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
    # Uncomment here for training again the model
    # r = TensorFlowRegressionModel(Config)
    # r.train(train_X, train_Y, LEARNING_RATE, TRAINING_EPOCHS, MODEL_DIR)
    # make some predictions with the stored model
    test_val = 6.83
    r = TensorFlowRegressionModel(Config, is_training=False)
    y_pred = r.predict(test_val)
    print y_pred
    assert y_pred > 2.48 and y_pred < 2.52

    return


if __name__ == "__main__":
    main()
