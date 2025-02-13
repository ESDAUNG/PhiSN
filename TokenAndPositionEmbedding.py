from tensorflow.keras import layers
import tensorflow as tf

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen=maxlen
        self.vocab_size=vocab_size
        self.embed_dim=embed_dim
        self.token_emb = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim,input_length=self.maxlen)
        self.pos_emb = layers.Embedding(input_dim=self.maxlen, output_dim=self.embed_dim)

    def call(self, x,training=False):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        return {"maxlen": self.maxlen, "vocab_size":self.vocab_size,"embed_dim":self.embed_dim}