
import tensorflow as tf
from models.embedding import embed_characters


class LogueModel:

    def __init__(self, params):
        L, d = params.sentence_len, params.hidden_size

        utters = tf.placeholder('int32', [None, L], 'input_utters')
        labels = tf.placeholder('int32', [None], 'labels')  # intent of an utterance (answer).

        utters_embed = embed_characters(utters, embed_dim=d)  # [None, L, D]

        with tf.variable_scope('Classifier'):
            logits = self.classify(utters_embed)

        with tf.variable_scope('Loss'):
            probs = tf.nn.sigmoid(logits)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(None, labels)

        self.utters, self.labels = utters, labels
        self.probs, self.losses = probs, losses

    def classify(self, embed):
        # TODO: implement
        pass
