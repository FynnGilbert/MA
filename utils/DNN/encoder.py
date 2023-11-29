#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
#from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
#from tensorflow.keras.layers import Add, Dense, Layer

class AttentionBase(tf.keras.layers.MultiHeadAttention):
    """
    args:
        num_heads: Number of attention heads.
        key_dim: Size of each attention head for query and key.
                 Dimensionality of the linearly projected queries and keys
        value_dim: Size of each attention head for value.
                   Dimensionality of the linearly projected values
    """
    def __init__(self, **kwargs):
        super().__init__(use_bias=True,**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()



class SelfAttention(AttentionBase):
    """
    Processes the 'context' sequence (input to the encoder)
    """
    def call(self, x, use_causal_mask: bool = False):
        attention, attention_scores = super().call(query=x,
                                                   value=x,
                                                   key=x,
                                                   use_causal_mask=use_causal_mask,
                                                   return_attention_scores=True,
                                                   )
        self.last_attention_scores = attention_scores  # cache for plotting
        return self.layer_norm(self.add([x, attention]))



class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model: int, dff: int, activation="relu", dropout: float = 0.1):
        super().__init__()
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation=activation),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()
    
    def call(self, x):
        return self.layer_norm(self.add([x, self.ffn(x)]))



class Encoder(tf.keras.layers.Layer):
    """Consists of a Positional Embedding and a stack of (self attention + ffn) models"""
    def __init__(self,
                 num_layers: int,
                 d_model: int,
                 dff: int,
                 num_heads: int,
                 activation: str = "relu",
                 dropout: float = 0.1,
                 name="TransformerEncoder"
                 ):
        if num_layers < 1:
            return ValueError("'num_layers' must be >= 1")
        super().__init__(name=name)
        self.sub_layers = []
        for _ in range(num_layers):
            self.sub_layers.append(
                tf.keras.Sequential([
                    SelfAttention(num_heads=num_heads, key_dim=d_model, value_dim=d_model, dropout=dropout),
                    FeedForward(activation=activation, d_model=d_model, dff=dff, dropout=dropout)
                ])
            )
        self.num_layers = num_layers
    
    def call(self, x):
        for layer in self.sub_layers:
            x = layer(x)
        return x

