from numpy import array
from tensorflow.keras import Model
from utils.DNN.model_layers import TransformerEncoder


def get_attention_scores(
        model: Model,
        model_inputs:list[array],
        layer_name: str="transformer_encoder_0"
    ) -> array:
    """
    returns attention score of transformer encoder layer
    """
    
    # get input and output layer
    layer = model.get_layer(layer_name)
    assert type(layer) == TransformerEncoder

    outputs = layer.output
    inputs=model.input
    
    # connect inputs with outputs
    intermediate_layer_model = Model(
        inputs=inputs,
        outputs=outputs
    )
    
    # get attention and turn it into an np.array
    _, attention = intermediate_layer_model(model_inputs)
    attention = attention.numpy()
    return attention

