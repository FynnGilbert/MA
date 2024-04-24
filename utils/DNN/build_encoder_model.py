#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from tensorflow.keras import layers
#from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
#from tensorflow.keras.layers import Add, Dense, Layer

@tf.keras.utils.register_keras_serializable()
class AddAcrossDimension(tf.keras.layers.Layer):
    """
    Performs Summation across the specified axis.

    Used to add all stack encoder representations together
    """

    def __init__(self, axis:int=1, **kwargs):
        super().__init__(**kwargs)

        self.axis=axis
        
        print("AddAcrossDimension has not been normalized yet")
        print("Consider Batch or LayerNormalization if the training becomes slow or unstable")

    def call(self, X):
        X = tf.math.reduce_sum(X, axis=self.axis, keepdims=False, name="ReduceStackDimension")
        return X

    def get_config(self): # 5
        config = super().get_config()
        config["axis"] = self.axis
        config["name"] = self.name # might cause Problem in case of double name
        return config




@tf.keras.utils.register_keras_serializable()
class AttentionBase(tf.keras.layers.MultiHeadAttention):
    """
    args:
        num_heads: Number of attention heads.
        key_dim: Size of each attention head for query and key.
                 Dimensionality of the linearly projected queries and keys
        value_dim: Size of each attention head for value.
                   Dimensionality of the linearly projected values
    """
    def __init__(self, norm:str="batch", seed:int=3093453, **kwargs):
        super().__init__(**kwargs)
        assert norm in ("batch", "layer")
        
        self.norm       = norm
        self.seed       = seed
        self.layer_norm = layers.LayerNormalization() if norm == "layer" else layers.BatchNormalization()
        self.add =        layers.Add()
    
    def get_config(self): # 5
        config = super().get_config()
        config["norm"] = self.norm
        config["seed"] = self.seed
        return config


@tf.keras.utils.register_keras_serializable()
class SelfAttention(AttentionBase):
    """
    Processes the 'context' sequence (input to the encoder)
    """
    def call(self, x):
        attention = self.layer_norm(x)
        attention, attention_scores = super().call(query=attention,
                                                   value=attention,
                                                   key=attention,
                                                   use_causal_mask=False,
                                                   return_attention_scores=True,
                                                   #seed=self.seed
                                                   )
        self.last_attention_scores = attention_scores  # cache for plotting

        return self.add([x, attention])

    def get_config(self): # 5
        config = super().get_config()
        return config


@tf.keras.utils.register_keras_serializable()
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model:int, dff:int, activation:str="relu", norm:str="batch", dropout:float=0.1, seed:int=3093453, **kwargs):
        super().__init__(**kwargs)
        
        assert norm in ("batch", "layer")
        self.d_model=d_model
        self.dff=dff
        self.activation=activation
        self.norm = norm
        self.dropout=dropout
        self.seed=seed

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation=activation),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout, seed=seed)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = layers.LayerNormalization() if norm == "layer" else layers.BatchNormalization()
         
    def call(self, x):
        return self.add([x, self.ffn(self.layer_norm(x))])
    
        #return self.add([x, self.ffn(x)])

    def get_config(self):
        config = super().get_config()
        config["d_model"] = self.d_model
        config["dff"] = self.dff
        config["activation"] = self.activation
        config["norm"] = self.norm
        config["dropout"] = self.dropout
        config["seed"] = self.seed

        return config


@tf.keras.utils.register_keras_serializable()
class Encoder(tf.keras.layers.Layer):
    """Consists of a Positional Embedding and a stack of (self attention + ffn) models"""
    def __init__(self,
                 num_layers:int,
                 num_heads:int,
                 d_model:int,
                 dff:int,
                 activation:str="relu",
                 norm:str="batch",
                 dropout:float=0.1,
                 seed:int=3093453,
                 **kwargs
                 ):
        if num_layers < 1:
            return ValueError("'num_layers' must be >= 1")

        assert norm in ("batch", "layer")
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.num_heads  = num_heads
        self.d_model    = d_model
        self.dff        = dff
        self.activation = activation
        self.norm       = norm
        self.dropout    = dropout
        self.seed       = seed

        self.sub_layers = []
        for _ in range(num_layers):
            self.sub_layers.append(
                tf.keras.Sequential([
                    SelfAttention(num_heads=num_heads, key_dim=d_model, value_dim=d_model, norm=norm, dropout=dropout,seed=seed),
                    FeedForward(activation=activation, d_model=d_model, dff=dff, norm=norm, dropout=dropout, seed=seed)
                ])
            )
        self.num_layers = num_layers
    
    def call(self, x):
        for layer in self.sub_layers:
            x = layer(x)
        return x


    def get_config(self):
        config = super().get_config()

        config["num_layers"] = self.num_layers
        config["num_heads"]  = self.num_heads
        config["d_model"]    = self.d_model
        config["dff"]        = self.dff
        config["activation"] = self.activation
        config["norm"]       = self.norm
        config["dropout"]    = self.dropout
        config["seed"]       = self.seed
        return config



@tf.keras.utils.register_keras_serializable()
class StackEncoder(tf.keras.layers.Layer):
    """Consists of a MaskingLayer, the Encoder, and StackAggregation"""
    def __init__(self,
                 num_layers:int,
                 num_heads:int,
                 d_model:int,
                 dff:int,
                 pad_size:int=80,
                 activation:str="relu",
                 norm:str="batch",
                 dropout:float=0.1,
                 seed:int=3093453,
                 **kwargs
                 ):
        if num_layers < 1:
            return ValueError("'num_layers' must be >= 1")
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.num_heads  = num_heads
        self.d_model    = d_model
        self.dff        = dff
        self.pad_size   = pad_size
        self.activation = activation
        self.norm       = norm
        self.dropout    = dropout
        self.seed       = seed


        self.masking = layers.Masking(input_shape=(pad_size, d_model))
        self.encoder = Encoder(num_layers=num_layers, num_heads=num_heads, d_model=d_model, dff=dff, norm=norm)
        self.add     = AddAcrossDimension(axis=1)


        self.stack_encoder = tf.keras.Sequential([
            #layers.Input(shape=(80, d_model)),
            self.masking,
            self.encoder,
            self.add,
        ])

    def call(self, x):
        return self.stack_encoder(x)

    def get_config(self):
        config = super().get_config()

        config["num_layers"] = self.num_layers
        config["num_heads"]  = self.num_heads
        config["d_model"]    = self.d_model
        config["dff"]        = self.dff
        config["pad_size"]   = self.pad_size
        config["activation"] = self.activation
        config["norm"]       = self.norm
        config["dropout"]    = self.dropout
        config["seed"]       = self.seed
        return config



def build_model(
        num_layers:int,
        num_heads:int,
        d_model:int,
        dff:int,
        labels:list[str]=["Improvement"],
        pad_size:int=80,
        activation:str="gelu",
        norm:str="batch",
        dropout:float=0.1,
        seed:int=3093453,
        
    ) -> tf.keras.Model:
    

    # inputs
    stack_input = layers.Input(shape=(pad_size, d_model))
    time_limit_input  = layers.Input(shape=(1,))
    inputs = [stack_input, time_limit_input]

    # stacks
    stack_encoder = StackEncoder(
            num_layers=num_layers, num_heads=num_heads, d_model=d_model, dff=dff,
            pad_size=pad_size, activation=activation, norm=norm, dropout=dropout
        )
    stack_encoder_out = stack_encoder(stack_input)

    # time_limit
    standardize_time_limit = layers.BatchNormalization()
    standardize_time_limit_out = standardize_time_limit(time_limit_input)

    # Concatenation
    concat_stacks_time = layers.Concatenate()
    concat_out = concat_stacks_time([stack_encoder_out, standardize_time_limit_out])

    dense_layer = layers.Dense(units=d_model, activation=activation)
    dense_layer_out = dense_layer(concat_out)

    # outputs
    
    ## Improvement
    improvement_prediction = layers.Dense(1, activation='sigmoid', name = "PredictionImprovement")
    improvement_prediction_out = improvement_prediction(dense_layer_out)

    ## Solved 
    solved_prediction = layers.Dense(1, activation='sigmoid', name = "PredictionSolved")
    solved_prediction_out = solved_prediction(dense_layer_out)

    ## AreaRatio
    area_prediction = layers.Dense(1, activation='sigmoid', name = "PredictionArea")
    area_prediction_out = area_prediction(dense_layer_out)

    ## stack prediction:
    stack_prediction = layers.Dense(pad_size, activation='sigmoid', name = "PredictionStacks")
    stack_prediction_out = stack_prediction(stack_encoder_out)

    ## Output selection
    output_layers = {
        "Improvement":  improvement_prediction_out,
        "Solved":       solved_prediction_out,
        "Area":         area_prediction_out,
        "Stacks":       stack_prediction_out,
    }
    outputs = [output_layers[label] for label in labels]

    model = tf.keras.Model(
        inputs = inputs,
        outputs=outputs
        )

    return model
