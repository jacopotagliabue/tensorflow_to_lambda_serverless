

class TensorFlowRegressionModel:

    def __init__(self, config, is_training=True):


        # if it is not training, restore the model and store the session in the class
        if not is_training:
            self.sess = self.restore_model_from_bucket()
        return

    def restore_model_from_bucket(self, bucket_name):


        return

    def train(self):

        return

    def upload_model_to_bucket(self, bucket_name):

        return

    def predict(self, x_val):

        return x_val