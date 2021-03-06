import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, TimeDistributed

from self_supervised_3d_tasks.algorithms.algorithm_base import AlgorithmBuilderBase
from self_supervised_3d_tasks.preprocessing.preprocess_simclr_weighted import preprocess_3d, get_patches_positions
from self_supervised_3d_tasks.utils.model_utils import apply_encoder_model_3d
import scipy

def get_euclidean_matrix():
    positions = get_patches_positions(4)

    return scipy.spatial.distance.cdist(positions,positions)

class SimclrWeightedBuilder(AlgorithmBuilderBase):
    def __init__(
            self,
            data_dim=384,
            number_channels=3,
            crop_size=None,
            patches_per_side=7,
            code_size=1024,
            lr=1e-3,
            data_is_3D=False,
            **kwargs,
    ):
        super(SimclrWeightedBuilder, self).__init__(data_dim, number_channels, lr, data_is_3D, **kwargs)

        self.patches_per_side = patches_per_side
        self.code_size = code_size
        self.number_channels = number_channels
        self.patches_number = patches_per_side * patches_per_side * patches_per_side

        self.patch_dim = int(self.data_dim / patches_per_side)
        self.patch_shape_3d = (self.patch_dim, self.patch_dim, self.patch_dim, self.number_channels)
        self.euclidean_dist = K.constant(get_euclidean_matrix())


    def apply_model(self):
        self.enc_model, _ = apply_encoder_model_3d(self.patch_shape_3d, **self.kwargs)

        return self.apply_prediction_model_to_encoder(self.enc_model)

    def apply_prediction_model_to_encoder(self, encoder_model):
        x_input = Input((self.patches_number, self.patch_dim, self.patch_dim, self.patch_dim, self.number_channels))
        patches_positions = Input((self.patches_number, 3))

        model_with_embed_dim = Sequential([encoder_model, Flatten(), Dense(self.code_size)])

        x_encoded = TimeDistributed(model_with_embed_dim)(x_input)

        simclr_weighted_model = keras.models.Model(inputs=[x_input, patches_positions], outputs=x_encoded)

        return simclr_weighted_model

    def l2_norm(self, x, axis=None):
        square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
        norm = K.sqrt(K.maximum(square_sum, K.epsilon()))

        return norm

    def simclr_weighted_loss(self, ytrue, ypredicted):
        """"
        A weighted version of the loss function presented in Simclr paper
        """
        predictions_norm = self.l2_norm(ypredicted, axis=2)

        transposed_predictions = K.permute_dimensions(ypredicted, (0,2,1))
        transposed_predictions_norm = self.l2_norm(transposed_predictions)

        dot_product = K.batch_dot(ypredicted, transposed_predictions)

        cosine_similarity = dot_product / (predictions_norm*transposed_predictions_norm)


        weighted_similarities = cosine_similarity *  self.euclidean_dist


        patch_loss = -K.log(K.exp(1.0)/ K.sum(K.exp(weighted_similarities) , axis = 2))

        return K.sum(patch_loss)

    def get_training_model(self):
        model = self.apply_model()
        model.compile(
            optimizer=keras.optimizers.Adam(lr=self.lr),
            loss=self.simclr_weighted_loss
        )

        return model

    def get_training_preprocessing(self):
        def f_3d(x, y):
            return preprocess_3d(x, self.patches_per_side)

        return f_3d, f_3d

    def get_finetuning_model(self, model_checkpoint=None):
        return super(SimclrWeightedBuilder, self).get_finetuning_model_patches(model_checkpoint)


def create_instance(*params, **kwargs):
    return SimclrWeightedBuilder(*params, **kwargs)
