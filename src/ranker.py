"""src/ranker.py - TF-Ranking neural listwise ranker."""

import tensorflow as tf

def approx_ndcg_loss(y_true, y_pred):
    """
    Differentiable ApproxNDCG loss (Qin et al. 2008).
    y_true: (batch, list_size) graded relevance, -1 = padding
    y_pred: (batch, list_size) scores
    """
    # Mask padding
    mask = tf.cast(y_true >= 0, tf.float32)
    y_true = y_true * mask

    # Approximate rank via softmax temperature scaling
    temperature = 0.1
    logits = y_pred / temperature
    # Subtract max for numerical stability
    logits = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
    approx_ranks = tf.reduce_sum(
        tf.sigmoid((logits[:, :, None] - logits[:, None, :])) * mask[:, None, :],
        axis=-1
    ) + 1.0

    # NDCG gain and discount
    gain     = (tf.pow(2.0, y_true) - 1.0) * mask
    discount = tf.math.log(1.0 + approx_ranks) / tf.math.log(2.0)
    dcg      = tf.reduce_sum(gain / discount, axis=-1)

    # Ideal DCG (sort true labels descending)
    ideal_sorted = tf.sort(y_true * mask, direction="DESCENDING", axis=-1)
    ideal_ranks  = tf.cast(tf.range(1, tf.shape(y_true)[1] + 1), tf.float32)
    ideal_gain   = tf.pow(2.0, ideal_sorted) - 1.0
    ideal_disc   = tf.math.log(1.0 + ideal_ranks) / tf.math.log(2.0)
    idcg         = tf.reduce_sum(ideal_gain / ideal_disc, axis=-1)

    # Normalize; avoid div-by-zero on all-padding groups
    ndcg = dcg / tf.maximum(idcg, 1e-6)
    return -tf.reduce_mean(ndcg)   # negate: we minimize loss

def build_scoring_network(hidden_dims: list[int], dropout: float, use_batch_norm: bool) -> tf.keras.Sequential:
    """
    Pointwise scoring network: (batch, list_size, n_features) -> (batch, list_size, 1)
    TF-Ranking applies this to each candidate independently, then computes listwise loss.
    """
    layers = []
    for dim in hidden_dims:
        if use_batch_norm:
            layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Dense(dim, activation="relu"))
        layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(1)) #scalar relevance score
    return tf.keras.Sequential(layers)

def build_ranker(n_features: int, hidden_dims: list[int], dropout: float, use_batch_norm: bool, learning_rate: float) -> tf.keras.Model:
    """
    Builds and compiles a TF-Ranking listwise model.
    Loss: ApproxNDCG - differentiable NDCG approximation.
    """

    scoring_network = build_scoring_network(hidden_dims, dropout, use_batch_norm)

    # Input: (batch, list_size, n_features)
    inputs = tf.keras.Input(shape=(None, n_features), name="features")
    scores = scoring_network(inputs)        # (batch, list_size, 1)
    scores = tf.keras.ops.squeeze(scores, axis=-1)    # (batch, list_size)
    model = tf.keras.Model(inputs=inputs, outputs=scores)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=approx_ndcg_loss # `tfr.keras.losses.ApproxNDCGLoss()` isn't available in this version of TF, so, use custom implementation, above.
        # metrics handled by eval_group() post training
    )

    return model

