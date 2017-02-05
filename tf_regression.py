# refactored from the examples at https://github.com/aymericdamien/TensorFlow-Examples
from s3_client import s3Client
import tensorflow as tf
import numpy
import matplotlib
matplotlib.use('TkAgg')  # avoid mac backend error
import matplotlib.pyplot as plt
import zipfile
import os


class TensorFlowRegressionModel:

    def __init__(self, config, is_training=True):
        # initialize s3 client to communicate with s3 buckets
        self.s3client = s3Client(config)
        # store the model variables into a class object
        self.vars = self.set_vars()
        self.model = self.build_model(self.vars)
        # if it is not training, restore the model and store the session in the class
        if not is_training:
            self.sess = self.restore_model_from_bucket()

        return

    def set_vars(self):
        """
        Define the linear regression model through the variables
        """
        return {
            # placeholders
            'X': tf.placeholder("float"),
            'Y': tf.placeholder("float"),
            # model weight and bias
            'W': tf.Variable(numpy.random.randn(), name="weight"),
            'b': tf.Variable(numpy.random.randn(), name="bias")
        }

    def build_model(self, vars):
        """
        Define the linear regression model through the variables
        """
        return tf.add(tf.mul(vars['X'], vars['W']), vars['b'])

    def restore_model_from_bucket(self, bucket_name):


        return

    def train(self, train_X, train_Y, learning_rate, training_epochs, model_output_dir=None, with_plot=False):
        n_samples = train_X.shape[0]
        # Mean squared error
        cost = tf.reduce_sum(tf.pow(self.model - self.vars['Y'], 2)) / (2 * n_samples)
        # Gradient descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        # Launch the graph
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            # Fit all training data
            for epoch in range(training_epochs):
                for x, y in zip(train_X, train_Y):
                    sess.run(optimizer, feed_dict={self.vars['X']: x, self.vars['Y']: y})
            # Print final metrics
            print "Epoch:", '%04d' % (epoch + 1), "W=", sess.run(self.vars['W']), "b=", sess.run(self.vars['b'])
            # Save model locally
            saver.save(sess, model_output_dir + 'model.ckpt')
            # Plot data if requested
            if with_plot:
                self.plot_data_vs_fitted(train_X, train_Y, sess.run(self.vars['W']) * train_X + sess.run(self.vars['b']))

        # Finally upload model to bucket
        self.upload_model_to_bucket('')

        return

    def upload_model_to_bucket(self, bucket_name):
        filenames = [os.path.join('model/', fn) for fn in next(os.walk('model/'))[2]]
        print filenames
        with zipfile.ZipFile("model/test.zip", "w") as z:
            for f in filenames:
                z.write(f)

        return

    def predict(self, x_val):

        return x_val

    def plot_data_vs_fitted(self, _x, _y, _fitted):
        plt.plot(_x, _y, 'ro', label='Original data')
        plt.plot(_x, _fitted, label='Fitted line')
        plt.legend()
        plt.show()

        return