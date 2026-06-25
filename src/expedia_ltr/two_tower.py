"""src/two_towers.py - TFRS two-tower retrieval model."""

import tensorflow as tf
import tensorflow_recommenders as tfrs

def build_tower(continuous_features: list[str],
                categorical_features: dict,
                tower_dims: list[int],
                dropout: float) -> tf.keras.Model:
    """
    Build one tower (query or item)
    Inputs: dict of feature tensors.
    Output: L2-normalized embedding of size tower_dims[-1].
    """

    inputs = {}
    parts = []

    # Continuous inputs - pass through directly
    for f in continuous_features:
        inp = tf.keras.Input(shape=(), name=f, dtype=tf.float32)
        inputs[f] = inp
        parts.append(tf.keras.layers.Reshape((1,))(inp))

    # Categorical inputs - embed
    for f, spec in categorical_features.items():
        inp = tf.keras.Input(shape=(), name=f, dtype=tf.int32)
        inputs[f] = inp
        emb = tf.keras.layers.Embedding(
            input_dim=spec["vocab_size"] + 1,    # +1 for OOV/padding
            output_dim=spec["embed_dim"],
            name=f"emb_{f}"
        )(inp)
        parts.append(emb)

    # Concatenate all parts
    x = tf.keras.layers.Concatenate()(parts) if len(parts) > 1 else parts[0]

    # MLP
    for dim in tower_dims[:-1]:
        x = tf.keras.layers.Dense(dim, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(tower_dims[-1])(x)  # no activation on final layer

    # L2 normalize so dot product = cosine similarity
    output = tf.keras.layers.Lambda(
        lambda v: tf.math.l2_normalize(v, axis=-1)
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=output)

class TwoTowerModel(tfrs.Model):
    """
    TFRS Model wrapping query and item towers.
    Loss: sampled softmax (in-batch negatives) via tfrs.tasks.Retrieval.
    """

    def __init__(self, query_tower, item_tower, item_dataset):
        super().__init__()
        self.query_tower = query_tower
        self.item_tower = item_tower

        # Retrieval task uses in-batch negatives by default
        self.task = tfrs.tasks.Retrieval()

    def compute_loss(self, inputs, training=False) -> tf.Tensor:
        query_emb = self.query_tower(inputs["query"], training=training)
        item_emb = self.item_tower(inputs["item"], training=training)
        return self.task(query_emb, item_emb)