#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.layers import Add, Dense, Layer


class TransformerEncoder(Layer):
    def __init__(
        self,
        units: int,
        num_heads: int,
        key_dim: int,
        idx: int,
        activation: str = "relu",
        dropout: float=0.05,
        use_bias: bool=False,
        bias_regularizer = None,
        use_PreLN: bool=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.self_attention_layer = MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim,
            dropout=dropout,               # Hyperparameter
            use_bias=use_bias,                     # usually False, but technically Hyperparameter
            bias_regularizer=bias_regularizer,
            name=f"Encoder-SelfAttentionLayer-{idx}"
        )
        
        self.add1 = Add(name=f"Encoder-1st-AdditionLayer-{idx}")
        self.add2 = Add(name=f"Encoder-2nd-AdditionLayer-{idx}")

        self.layernorm1 = LayerNormalization(name=f"Encoder-1st-NormalizationLayer-{idx}")
        self.layernorm2 = LayerNormalization(name=f"Encoder-2nd-NormalizationLayer-{idx}")

        self.feed_forward_layer = Dense(
            units=units,
            activation=activation,
            name=f"Encoder-FeedForwardLayer_{idx}"
        )

        # Actual Parameters:
        self.use_PreLN = use_PreLN

    def call(self, inputs):
        """
        Check the paper 'On Layer Normalization in the Transformer Architecture'
        """

        #### Attention Block ####
        
        attention_inputs = inputs

        if (self.use_PreLN is not None) and self.use_PreLN:
            attention_inputs = self.layernorm1(attention_inputs)

        self_attention_output, attention_scores = self.self_attention_layer(
            query=attention_inputs,
            value=attention_inputs,
            key = attention_inputs,
            use_causal_mask=False, # only makes sense for Time series / causal flow data
            return_attention_scores=True
        )

        inputs = self.add1([inputs, self_attention_output])

        if (self.use_PreLN is not None) and not self.use_PreLN:
            inputs = self.layernorm1(inputs)

        #### Feed Forward Block ####

        ff_inputs = inputs

        if (self.use_PreLN is not None) and self.use_PreLN:
            ff_inputs = self.layernorm2(ff_inputs)

        feed_forward_output = self.feed_forward_layer(ff_inputs)

        inputs = self.add2([inputs, feed_forward_output])

        if (self.use_PreLN is not None) and not self.use_PreLN:
            inputs = self.layernorm2(inputs)

        return inputs, attention_scores 
