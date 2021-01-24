import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.layers import Flatten, TimeDistributed

from self_supervised_3d_tasks.algorithms.algorithm_base import AlgorithmBuilderBase
from self_supervised_3d_tasks.preprocessing.preprocess_simclr import preprocess_3d_batch_level_loss, preprocess_3d_volume_level_loss
from self_supervised_3d_tasks.utils.model_utils import apply_encoder_model_3d

class SimclrBuilder(AlgorithmBuilderBase):
    def __init__(
            self,
            data_dim=384,
            number_channels=3,
            crop_size=None,
            patches_per_side=7,
            code_size=1024,
            lr=1e-3,
            data_is_3D=False,
            temprature=0.05,
            augmentations=[],
            loss_function_name='contrastive_loss_volume_level',
            **kwargs,
    ):
        super(SimclrBuilder, self).__init__(data_dim, number_channels, lr, data_is_3D, **kwargs)

        self.loss_function = self.contrastive_loss_volume_level
        if loss_function_name =='contrastive_loss_batch_level':
            self.loss_function = self.contrastive_loss_batch_level

        self.temprature = temprature
        self.augmentations = augmentations
        self.patches_per_side = patches_per_side
        self.code_size = code_size
        self.number_channels = number_channels
        self.patches_number = patches_per_side * patches_per_side * patches_per_side * 2

        self.patch_dim = int(self.data_dim / patches_per_side)
        self.patch_shape_3d = (self.patch_dim, self.patch_dim, self.patch_dim, self.number_channels)

        self.inverse_eye = 1 - K.eye(self.patches_number)
        self.inverse_eye = K.expand_dims(self.inverse_eye, 0)

        # This mask will be used to extract the numerator part of the loss function
        self.numerator_mask =  np.zeros(shape=(1, self.patches_number, self.patches_number))

        for i in range(1,self.patches_number,2):
            self.numerator_mask[0, i, i-1] = 1
            self.numerator_mask[0, i-1, i] = 1

    def apply_model(self):
        self.enc_model, _ = apply_encoder_model_3d(self.patch_shape_3d, **self.kwargs)

        return self.apply_prediction_model_to_encoder(self.enc_model)

    def apply_prediction_model_to_encoder(self, encoder_model):
        x_input = Input((self.patches_number, self.patch_dim, self.patch_dim, self.patch_dim, self.number_channels))

        model_with_embed_dim = Sequential([encoder_model, Flatten(), Dense(self.code_size)])

        x_encoded = TimeDistributed(model_with_embed_dim)(x_input)

        x_encoded = Lambda(self.reshape_predictions)(x_encoded)

        simclr_model = keras.models.Model(inputs=x_input, outputs=x_encoded)

        return simclr_model

    def reshape_predictions(self, predictions):
        # Only reshape when we want to compute the contrastive loss on batch level
        if self.loss_function == self.contrastive_loss_volume_level:
            return predictions

        mid_index = int(predictions.shape[1]/2)

        predictions_1 = predictions[:,:mid_index,:]
        predictions_1 = tf.reshape(predictions_1, (-1, predictions_1.shape[2]))

        predictions_2 = predictions[:,mid_index:,:]
        predictions_2 = tf.reshape(predictions_2, (-1, predictions_2.shape[2]))

        return tf.concat((predictions_1, predictions_2), axis=0)

    def l2_norm(self, x, axis=None):
        square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
        norm = K.sqrt(K.maximum(square_sum, K.epsilon()))
        return norm

    def contrastive_loss_volume_level(self, ytrue, ypredicted):
        #predictions_shape = K.print_tensor(K.shape(ypredicted))
        predictions_norm = self.l2_norm(ypredicted, axis=2)

        transposed_predictions = K.permute_dimensions(ypredicted, (0,2,1))
        transposed_predictions_norm = self.l2_norm(transposed_predictions, axis=1)

        norms = K.batch_dot(predictions_norm, transposed_predictions_norm)

        dot_product = K.batch_dot(ypredicted, transposed_predictions)
        cosine_similarity = dot_product / norms

        # Set self similarity to zero so that we can calculate losses through matrix operations
        similarities = K.exp(cosine_similarity / self.temprature)
        similarities = similarities * self.inverse_eye

        denominator = K.sum(similarities, axis=2)

        # By multiplying with the mask and then summing on axis 2 we are effectivly just selecting the pairs
        # As suggested in the contrastive loss function E.g.
        # For patch 1 we only keep patch 2 and vice versa
        # For patch 3 we only keep patch 4 and vice versa
        numerator = similarities * self.numerator_mask
        numerator = K.sum(numerator, axis=2)

        batch_loss = -K.log(numerator / denominator)

        return K.mean(batch_loss)

    def contrastive_loss_batch_level(self, ytrue, ypredicted):
        #predictions_shape = K.print_tensor(K.shape(ypredicted))
        patches_number = K.shape(ypredicted)[0]

        predictions_norm = self.l2_norm(ypredicted, axis=1)

        transposed_predictions = K.permute_dimensions(ypredicted, (1, 0))
        transposed_predictions_norm = self.l2_norm(transposed_predictions, axis=0)

        norms = K.dot(predictions_norm, transposed_predictions_norm)

        dot_product = K.dot(ypredicted, transposed_predictions)
        cosine_similarity = dot_product / norms

        # Set self similarity to zero so that we can calculate losses through matrix operations
        similarities = K.exp(cosine_similarity / self.temprature)
        similarities_mask = 1 - tf.one_hot(tf.range(patches_number), patches_number)
        similarities = similarities * similarities_mask

        # Calculate denominator
        denominator = K.sum(similarities, axis=1)

        # Calculate numerator
        mid_index = tf.cast(tf.math.divide(patches_number, 2), tf.int32)

        # Because of the way we reshaped the predictions in. The similarities between
        # two augementations of the same patch can be found at the main diagonal of the
        # second quarter and thrid quarter of the similarities matrix.
        # I.E., if we divide the sqaured shape similarities matrix into 4 quarters, then the main diagonals
        # of the second quarter ( (0 -> mid_index) rows, (mid_index -> len(similarities)) columns)
        # and the third quarter ( (mid_index -> len(similarities)) rows, (0 -> mid_index) columns)
        similarities_1 = tf.slice(similarities, [0,mid_index], [mid_index, mid_index])
        similarities_1 = tf.linalg.diag_part(similarities_1)

        similarities_2 = tf.slice(similarities, [mid_index,0], [mid_index, mid_index])
        similarities_2 = tf.linalg.diag_part(similarities_2)

        numerator = tf.concat((similarities_1, similarities_2), axis=0)

        # Final Loss
        return K.mean( -K.log(numerator / denominator))

    def get_training_model(self):
        model = self.apply_model()
        model.compile(
            optimizer=keras.optimizers.Adam(lr=self.lr),
            loss=self.loss_function,
            metrics=[self.loss_function]
        )
        return model

    def get_training_preprocessing(self):
        def f_3d(x, y):
            if self.loss_function == self.contrastive_loss_volume_level:
                return preprocess_3d_volume_level_loss(x, self.patches_per_side, self.augmentations)

            return preprocess_3d_batch_level_loss(x, self.patches_per_side, self.augmentations)

        return f_3d, f_3d

    def get_finetuning_model(self, model_checkpoint=None):
        return super(SimclrBuilder, self).get_finetuning_model_patches(model_checkpoint)


def create_instance(*params, **kwargs):
    return SimclrBuilder(*params, **kwargs)
