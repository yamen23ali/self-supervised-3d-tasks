import numpy as np
import math
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Lambda, Reshape
from tensorflow.keras.layers import Flatten, TimeDistributed, multiply, UpSampling3D
from tensorflow.keras.losses import mse
from self_supervised_3d_tasks.algorithms.algorithm_base import AlgorithmBuilderBase
from self_supervised_3d_tasks.preprocessing.preprocess_simclr import preprocess_3d_batch_level_loss, preprocess_3d_volume_level_loss
from self_supervised_3d_tasks.utils.model_utils import apply_encoder_model_3d, print_flat_summary, apply_decoder_model_from_encoder
from self_supervised_3d_tasks.models.unet3d import downconv_model_3d, upconv_model_3d

class SimclrBuilder(AlgorithmBuilderBase):
    def __init__(
            self,
            data_dim=384,
            data_dim_z=384,
            patches_per_side=3,
            number_channels=3,
            crop_size=None,
            patches_in_depth=7,
            code_size=1024,
            lr=1e-3,
            data_is_3D=False,
            temprature=0.05,
            augmentations=[],
            loss_function_name='contrastive_loss_volume_level',
            position_based_mask = False,
            data_cropped = False,
            contrastive_loss_weight = 0.5,
            reconstruction_loss_weight = 0.5,
            **kwargs,
    ):
        super(SimclrBuilder, self).__init__(data_dim, number_channels, lr, data_is_3D, **kwargs)

        self.contrastive_loss_function = self.contrastive_loss_volume_level
        if loss_function_name =='contrastive_loss_batch_level':
            self.contrastive_loss_function = self.contrastive_loss_batch_level

        self.temprature = temprature
        self.augmentations = augmentations
        self.code_size = code_size
        self.number_channels = number_channels
        self.position_based_mask = position_based_mask
        self.patches_in_depth = patches_in_depth
        self.patches_per_side = patches_per_side
        self.data_cropped = data_cropped
        self.embeddings_output = 'EMBEDDINGS_OUTPUT'
        self.decoder_output = 'DECODER_OUTPUT'
        self.contrastive_loss_weight = contrastive_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight

        if data_cropped:
            self.patches_number = patches_in_depth * 2
            depth_dim = int(data_dim_z / patches_in_depth)
            self.patch_shape_3d = (data_dim, data_dim, depth_dim, self.number_channels)
            self.input_shape = (self.patches_number, data_dim, data_dim, depth_dim, self.number_channels)
        else:
            self.patches_number = patches_per_side * patches_per_side * patches_per_side * 2
            self.patch_dim = int(self.data_dim / patches_per_side)
            self.patch_shape_3d = (self.patch_dim, self.patch_dim, self.patch_dim, self.number_channels)
            self.input_shape = (self.patches_number, self.patch_dim, self.patch_dim, self.patch_dim, self.number_channels)

        self.inverse_eye = 1 - K.eye(self.patches_number)
        self.inverse_eye = K.expand_dims(self.inverse_eye, 0)

        # This mask will be used to extract the numerator part of the loss function
        self.numerator_mask =  np.zeros(shape=(1, self.patches_number, self.patches_number))

        for i in range(1,self.patches_number,2):
            self.numerator_mask[0, i, i-1] = 1
            self.numerator_mask[0, i-1, i] = 1

    def apply_model(self):
        self.enc_model, enc_layer_data = apply_encoder_model_3d(self.patch_shape_3d, **self.kwargs)
        self.dec_model = apply_decoder_model_from_encoder(self.enc_model, enc_layer_data)

        return self.apply_prediction_model_to_encoder(self.enc_model, self.dec_model)

    def apply_prediction_model_to_encoder(self, encoder_model, decoder_model):
        x_input = Input(self.input_shape)

        embeddings_model = Sequential([Flatten(), Dense(self.code_size)])
        x_encoded = TimeDistributed(encoder_model)(x_input)

        # Contrastive output
        embeddings = TimeDistributed(embeddings_model)(x_encoded)
        contrastive_output = Lambda(
            self.reshape_predictions, name=self.embeddings_output)(embeddings)

        # Enc Dec output
        enc_dec_output = TimeDistributed(decoder_model,  name=self.decoder_output)(x_encoded)

        simclr_model = keras.models.Model(
            x_input, [contrastive_output, enc_dec_output])

        return simclr_model

    def reconstruction_loss(self, ytrue, ypredicted):
        #if self.contrastive_loss_function == self.contrastive_loss_volume_level:
        return K.mean(mse(ytrue[:,::2], ypredicted[:,1::2])) + K.mean(mse(ytrue[:,1::2], ypredicted[:,::2]))
        #return K.mean(mse(ytrue[::2], ypredicted[1::2])) + K.mean(mse(ytrue[1::2], ypredicted[::2]))


    def reshape_predictions(self, predictions):
        #K.print_tensor(K.shape(predictions))
        # Only reshape when we want to compute the contrastive loss on batch level
        if self.contrastive_loss_function == self.contrastive_loss_volume_level:
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
        #print("======= Batch ======")
        #K.print_tensor(K.shape(ypredicted))
        #K.print_tensor(K.shape(ytrue))
        patches_number = K.shape(ypredicted)[0]
        #K.print_tensor(patches_number)

        predictions_norm = self.l2_norm(ypredicted, axis=1)

        transposed_predictions = K.permute_dimensions(ypredicted, (1, 0))
        transposed_predictions_norm = self.l2_norm(transposed_predictions, axis=0)

        norms = K.dot(predictions_norm, transposed_predictions_norm)

        dot_product = K.dot(ypredicted, transposed_predictions)
        cosine_similarity = dot_product / norms

        # Set self similarity to zero so that we can calculate losses through matrix operations
        similarities = K.exp(cosine_similarity / self.temprature)
        identity_mask = 1 - tf.one_hot(tf.range(patches_number), patches_number)
        similarities = similarities * identity_mask
        #K.print_tensor(K.shape(similarities))

        # Calculate denominator
        denominator = None
        if self.position_based_mask:
            # A mask that mark all pairs in similar positions with 0
            # The idea here is to not enforce the model to
            # consider pairs in similar positions either (similar nor dissimilar)
            mask_shape = tf.cast(tf.math.sqrt(tf.cast(K.shape(ytrue)[1], tf.float32)), tf.int32)
            similarities_mask = Reshape((mask_shape, mask_shape))(ytrue)[0]
            denominator_similarities = similarities * similarities_mask
            denominator = K.sum(denominator_similarities, axis=1)
        else:
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
            loss={
                self.embeddings_output: self.contrastive_loss_function,
                self.decoder_output: self.reconstruction_loss
            },
            loss_weights={
                self.embeddings_output: self.contrastive_loss_weight,
                self.decoder_output: self.reconstruction_loss_weight
            },
            metrics={
                self.embeddings_output: self.contrastive_loss_function,
                self.decoder_output: self.reconstruction_loss
            }
        )
        return model

    def get_training_preprocessing(self):
        def simclr_f_3d(x, y, files_names):
            if self.contrastive_loss_function == self.contrastive_loss_volume_level:
                return preprocess_3d_volume_level_loss(
                    x,
                    self.augmentations,
                    data_cropped=self.data_cropped,
                    patches_in_depth=self.patches_in_depth,
                    patches_per_side=self.patches_per_side)

            return preprocess_3d_batch_level_loss(
                x,
                self.augmentations,
                files_names,
                self.position_based_mask,
                data_cropped=self.data_cropped,
                patches_in_depth=self.patches_in_depth,
                patches_per_side=self.patches_per_side)

        return simclr_f_3d, simclr_f_3d

    def get_finetuning_model(self, model_checkpoint=None):
        return super(SimclrBuilder, self).get_finetuning_model_patches(model_checkpoint)

    def get_finetuning_model_with_dec(self, model_checkpoint=None):
        return super(SimclrBuilder, self).get_finetuning_model_with_dec_patches(model_checkpoint)


def create_instance(*params, **kwargs):
    return SimclrBuilder(*params, **kwargs)
