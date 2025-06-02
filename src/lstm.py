import os
import random
import time
import pickle
import gc
import logging
import warnings

os.environ['PYTHONHASHSEED']       = '3888'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

random.seed(3888)
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

import numpy as np
import pandas as pd
np.random.seed(3888)

import tensorflow as tf
tf.random.set_seed(3888)

from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Bidirectional,
    Dropout,
    LayerNormalization,
    Dense,
    Lambda,
    Multiply,
    Add,
    Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

import src.util as util

def baseline(
    snapshot_df: pd.DataFrame,
    basic_features: list = ['wap', 'log_return'],
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    es_patience: int = 10,
    lr_patience: int = 5,
    model_name: str = 'baseline'
):

    seq_df = util.generate_tick_sequences(snapshot_df, feature_cols=basic_features)

    # Split by time_id
    unique_tids = np.sort(seq_df['time_id'].unique())
    split_idx = int(len(unique_tids) * 0.8)
    train_tids, val_tids = unique_tids[:split_idx], unique_tids[split_idx:]

    train_df = seq_df[seq_df['time_id'].isin(train_tids)].reset_index(drop=True)
    val_df   = seq_df[seq_df['time_id'].isin(val_tids)].reset_index(drop=True)

    # Fit scalers on training data
    flat_X_train = np.vstack(train_df['X'].values).astype('float32')
    x_scaler = StandardScaler().fit(flat_X_train)
    y_scaler = StandardScaler().fit(train_df['y'].values.reshape(-1,1))

    def train_generator():
        while True:
            for i in range(0, len(train_df), batch_size):
                b = train_df.iloc[i:i+batch_size]
                Xb = np.stack(b['X']).astype('float32')
                yb = b['y'].values.astype('float32').reshape(-1,1)
                yield Xb, yb

    def val_generator():
        while True:
            for i in range(0, len(val_df), batch_size):
                b = val_df.iloc[i:i+batch_size]
                Xb = np.stack(b['X']).astype('float32')
                yb = b['y'].values.astype('float32').reshape(-1,1)
                yield Xb, yb

    steps_per_epoch = len(train_df) // batch_size
    validation_steps = len(val_df) // batch_size

    # Build model architecture
    W, D = train_df.iloc[0]['X'].shape
    inputs = Input((W, D))
    x = LSTM(64, name='lstm_layer')(inputs)
    outputs = Dense(1, name='output')(x)
    model = Model(inputs, outputs, name='Baseline_LSTM')
    model.compile(optimizer=Adam(learning_rate), loss='mse')

    # Training callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=es_patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=lr_patience, min_lr=1e-6)
    ]

    # Fit model
    history = model.fit(
        train_generator(),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator(),
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=0
    )

    # Inference on validation set
    inference_times, preds = [], []
    X_val_flat = np.vstack(val_df['X'].values).astype('float32')
    X_val_scaled = x_scaler.transform(X_val_flat).reshape(-1, W, D)
    y_true = val_df['y'].values
    for seq in X_val_scaled:
        t0 = time.perf_counter()
        pred = model.predict(seq[np.newaxis,...], verbose=0).ravel()[0]
        inference_times.append(time.perf_counter() - t0)
        preds.append(pred)
    preds = np.array(preds)
    y_pred = y_scaler.inverse_transform(preds.reshape(-1,1)).ravel()

    results = val_df[['time_id','start_time']].copy()
    results['y_true'] = y_true
    results['y_pred'] = y_pred
    results['inference_time'] = inference_times

    # 11. Save model and scalers
    out_dir = os.path.join('models','lstm')
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, f"{model_name}.h5"), save_format='tf')
    with open(os.path.join(out_dir, f"{model_name}_scalers.pkl"), 'wb') as f:
        pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)

    tf.keras.backend.clear_session()
    del model
    gc.collect()

    return history, results



# Custom weighted MSE loss function
def weighted_mse(y_true, y_pred):
    """
    Weighted mean squared error: larger true values have higher weight.
    """
    abs_y = tf.abs(y_true)
    denom = tf.reduce_mean(abs_y) + 1e-6
    w = 1.0 + abs_y / denom
    return tf.reduce_mean(w * tf.square(y_true - y_pred))


def baseline_weighted_mse(
    snapshot_df: pd.DataFrame,
    basic_features: list = ['wap', 'log_return'],
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    es_patience: int = 10,
    lr_patience: int = 5,
    model_name: str = 'baseline_weighted_mse'
):

    # 1. Build sequences and split by time_id
    seq_df = util.generate_tick_sequences(snapshot_df, feature_cols=basic_features)
    tids = np.sort(seq_df['time_id'].unique())
    split_idx = int(len(tids) * 0.8)
    train_df = seq_df[seq_df['time_id'].isin(tids[:split_idx])].reset_index(drop=True)
    val_df   = seq_df[seq_df['time_id'].isin(tids[split_idx:])].reset_index(drop=True)

    # 2. Fit scalers on training data
    X_flat = np.vstack(train_df['X'].values).astype('float32')
    x_scaler = StandardScaler().fit(X_flat)
    y_scaler = StandardScaler().fit(train_df['y'].values.reshape(-1,1))

    def train_generator():
        while True:
            for i in range(0, len(train_df), batch_size):
                b = train_df.iloc[i:i+batch_size]
                Xb = np.stack(b['X']).astype('float32')
                yb = b['y'].values.astype('float32').reshape(-1,1)
                yield Xb, yb

    def val_generator():
        while True:
            for i in range(0, len(val_df), batch_size):
                b = val_df.iloc[i:i+batch_size]
                Xb = np.stack(b['X']).astype('float32')
                yb = b['y'].values.astype('float32').reshape(-1,1)
                yield Xb, yb

    steps_per_epoch  = len(train_df) // batch_size
    validation_steps = len(val_df)   // batch_size

    # 4. Build and compile model
    W, D = train_df.iloc[0]['X'].shape
    inputs = Input((W, D))
    x = LSTM(64)(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs, name='LSTM_WeightedMSE')
    model.compile(optimizer=Adam(learning_rate), loss=weighted_mse)

    # 5. Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=es_patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=lr_patience, min_lr=1e-6)
    ]

    # 6. Train
    history = model.fit(
        train_generator(),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator(),
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=0
    )

    # 7. Inference on validation set
    inference_times, preds = [], []
    X_val_flat = np.vstack(val_df['X'].values).astype('float32')
    X_val_scaled = x_scaler.transform(X_val_flat).reshape(-1, W, D)
    y_true = val_df['y'].values
    for seq in X_val_scaled:
        t0 = time.perf_counter()
        p = model.predict(seq[np.newaxis,...], verbose=0).ravel()[0]
        inference_times.append(time.perf_counter() - t0)
        preds.append(p)
    y_pred = y_scaler.inverse_transform(np.array(preds).reshape(-1,1)).ravel()

    # 8. Compile results
    results = val_df[['time_id','start_time']].copy()
    results['y_true'] = y_true
    results['y_pred'] = y_pred
    results['inference_time'] = inference_times

    # 10. Save model and scalers
    out_dir = os.path.join('models','lstm')
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, f"{model_name}.h5"), save_format='tf')
    with open(os.path.join(out_dir, f"{model_name}_scalers.pkl"), 'wb') as f:
        pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)

    tf.keras.backend.clear_session()
    del model
    gc.collect()

    return history, results



class WarmUp(tf.keras.callbacks.Callback):
    def __init__(self, warmup_steps, target_lr):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr

    def on_train_batch_begin(self, batch, logs=None):
        it = self.model.optimizer.iterations.numpy()
        if it < self.warmup_steps:
            warmup_lr = self.target_lr * (it / float(self.warmup_steps))
            try:
                self.model.optimizer.learning_rate = warmup_lr
            except AttributeError:
                self.model.optimizer._set_hyper('learning_rate', warmup_lr)


def moe(
    snapshot_df: pd.DataFrame,
    basic_features: list,
    spike_quantile: float = 0.90,
    target_frac: float = 0.20,
    min_factor: float = 1.0,
    max_factor: float = 5.0,
    temperature: float = 0.3,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    es_patience: int = 10,
    lr_patience: int = 5,
    model_name: str = 'moe'
) -> tuple:

    seq_df = util.generate_tick_sequences(snapshot_df, feature_cols=basic_features)
    tids = np.sort(seq_df['time_id'].unique())
    split = int(len(tids) * 0.8)
    train_df = seq_df[seq_df['time_id'].isin(tids[:split])].reset_index(drop=True)
    val_df   = seq_df[seq_df['time_id'].isin(tids[split:])].reset_index(drop=True)

    spike_thresh = train_df['y'].quantile(spike_quantile)
    train_df['spike'] = (train_df['y'] > spike_thresh).astype(int)
    val_df['spike']   = (val_df['y']   > spike_thresh).astype(int)

    flat_X = np.vstack(train_df['X'].values).astype('float32')
    x_scaler = StandardScaler().fit(flat_X)
    y_scaler = StandardScaler().fit(train_df['y'].values.reshape(-1,1))

    pos = train_df['spike'].sum()
    neg = len(train_df) - pos
    raw = (target_frac * neg / (pos * (1 - target_frac))) if pos>0 else 1.0
    factor = int(np.clip(raw, min_factor, max_factor))

    def train_generator():
        """Infinite training batch generator with oversampling."""
        pos_df = train_df[train_df['spike']==1]
        neg_df = train_df[train_df['spike']==0]
        while True:
            neg_shuf = neg_df.sample(frac=1).reset_index(drop=True)
            pos_shuf = pos_df.sample(frac=factor, replace=True).reset_index(drop=True)
            batch_df = pd.concat([neg_shuf, pos_shuf]).sample(frac=1).reset_index(drop=True)
            for i in range(0, len(batch_df), batch_size):
                batch = batch_df.iloc[i:i+batch_size]
                Xb = np.stack(batch['X'].values).astype('float32')
                Xb_s = x_scaler.transform(Xb.reshape(-1, Xb.shape[-1])).reshape(Xb.shape)
                yb = batch['y'].values.reshape(-1,1).astype('float32')
                yb_s = y_scaler.transform(yb).ravel()
                spk = batch['spike'].values.astype('float32').reshape(-1,1)
                yield Xb_s, {'vol': yb_s, 'spike': spk}

    def val_generator():
        """Infinite validation batch generator."""
        W, D = train_df.iloc[0]['X'].shape
        while True:
            for i in range(0, len(val_df), batch_size):
                batch = val_df.iloc[i:i+batch_size]
                Xb = np.stack(batch['X'].values).astype('float32')
                Xb_s = x_scaler.transform(Xb.reshape(-1, D)).reshape(-1, W, D)
                yb_s = y_scaler.transform(batch['y'].values.reshape(-1,1)).ravel()
                spk = batch['spike'].values.astype('float32').reshape(-1,1)
                yield Xb_s, {'vol': yb_s, 'spike': spk}

    steps  = len(train_df) // batch_size
    vsteps = int(np.ceil(len(val_df) / batch_size))

    W, D = train_df.iloc[0]['X'].shape

    inp           = Input(shape=(W, D), name="input_sequence")
    h             = LSTM(128, name="lstm")(inp)
    spike_logits  = Dense(1, name="spike_logits")(h)
    spike_pred    = Lambda(
        lambda t: tf.math.sigmoid(t / temperature),
        output_shape=lambda s: s,
        name="spike"
    )(spike_logits)

    expert_norm   = Dense(1, name="expert_normal")(h)
    expert_spike  = Dense(1, name="expert_spike")(h)

    one_minus_spike = Lambda(
        lambda t: 1.0 - t,
        output_shape=lambda s: s,
        name="one_minus_spike"
    )(spike_pred)

    norm_contrib  = Multiply(name="norm_contrib")([one_minus_spike, expert_norm])
    spike_contrib = Multiply(name="spike_contrib")([spike_pred, expert_spike])
    vol_pred      = Add(name="vol")([norm_contrib, spike_contrib])

    model = Model(inputs=inp, outputs=[vol_pred, spike_pred])



    def weighted_mse(y_true, y_pred):
        abs_y = K.abs(y_true)
        w = 1.0 + abs_y / (K.mean(abs_y) + K.epsilon())
        return K.mean(w * K.square(y_true - y_pred))

    def focal_bce(gamma=2.0, alpha=0.5):
        bce = tf.keras.losses.BinaryCrossentropy()
        def loss(y_true, y_pred):
            p_t = y_true*y_pred + (1-y_true)*(1-y_pred)
            return K.mean(alpha * K.pow(1-p_t, gamma) * bce(y_true, y_pred))
        return loss

    model.compile(
        optimizer=Adam(learning_rate),
        loss={'vol': weighted_mse, 'spike': focal_bce()},
        loss_weights={'vol':1.0, 'spike':0.5},
        metrics={'vol': tf.keras.metrics.RootMeanSquaredError(), 'spike': 'accuracy'}
    )

    warmup_steps = steps
    warmup_cb = WarmUp(warmup_steps, learning_rate)
    es = EarlyStopping(
        monitor='val_vol_loss',
        mode='min',
        patience=es_patience,
        restore_best_weights=True,
        verbose=0
    )
    rlr = ReduceLROnPlateau(
        monitor='val_vol_loss',
        mode='min',
        factor=0.5,
        patience=lr_patience,
        min_lr=1e-7,
        verbose=0
    )

    history = model.fit(
        train_generator(),
        validation_data=val_generator(),
        steps_per_epoch=steps,
        validation_steps=vsteps,
        epochs=epochs,
        callbacks=[warmup_cb, es, rlr],
        verbose=0
    )

    inf = Model(inp, [vol_pred, spike_pred, expert_norm, expert_spike])
    inf.set_weights(model.get_weights())

    flat_val = np.vstack(val_df['X'].values).astype('float32')
    Xv = x_scaler.transform(flat_val).reshape(-1, W, D)
    y_true = val_df['y'].values
    vols, probs, norms, spks, times = [], [], [], [], []
    for seq in Xv:
        t0 = time.perf_counter()
        v, p, n, s = inf.predict(seq[np.newaxis], verbose=0)
        times.append(time.perf_counter() - t0)
        vols.append(v.ravel()[0]); probs.append(p.ravel()[0])
        norms.append(n.ravel()[0]); spks.append(s.ravel()[0])

    results = val_df[['time_id','start_time']].copy()
    results['y_true']        = y_true
    results['y_pred']        = y_scaler.inverse_transform(np.array(vols).reshape(-1,1)).ravel()
    results['spike_prob']    = probs
    results['expert_normal'] = y_scaler.inverse_transform(np.array(norms).reshape(-1,1)).ravel()
    results['expert_spike']  = y_scaler.inverse_transform(np.array(spks).reshape(-1,1)).ravel()
    results['inference_time'] = times

    out_dir = os.path.join('models','lstm')
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, f"{model_name}.h5"), save_format='tf')
    with open(os.path.join(out_dir, f"{model_name}_scalers.pkl"), 'wb') as f:
        pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)

    tf.keras.backend.clear_session()
    del model, inf
    gc.collect()

    return history, results


def moe_staged(
    snapshot_df,
    basic_features,
    spike_quantile: float = 0.90,
    target_frac: float = 0.20,
    min_factor: float = 1.0,
    max_factor: float = 5.0,
    temperature: float = 0.3,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    es_patience: int = 10,
    lr_patience: int = 5,
    model_name: str = 'moe_staged'
) -> tuple:
    
    # --- Prepare data ---
    seq_df = util.generate_tick_sequences(snapshot_df, feature_cols=basic_features)
    tids = np.sort(seq_df['time_id'].unique())
    split = int(len(tids) * 0.8)
    train_df = seq_df[seq_df['time_id'].isin(tids[:split])].reset_index(drop=True)
    val_df   = seq_df[seq_df['time_id'].isin(tids[split:])].reset_index(drop=True)

    spike_thresh = train_df['y'].quantile(spike_quantile)
    train_df['spike'] = (train_df['y'] > spike_thresh).astype(int)
    val_df['spike']   = (val_df['y'] > spike_thresh).astype(int)

    # --- Scaling ---
    flat_X = np.vstack(train_df['X'].values).astype('float32')
    x_scaler = StandardScaler().fit(flat_X)
    y_scaler = StandardScaler().fit(train_df['y'].values.reshape(-1,1))

    # --- Generators ---
    def train_generator():
        pos_df = train_df[train_df['spike']==1]
        neg_df = train_df[train_df['spike']==0]
        while True:
            neg_shuf = neg_df.sample(frac=1).reset_index(drop=True)
            raw = (target_frac * len(neg_df) / (len(pos_df)*(1-target_frac))) if len(pos_df)>0 else 1.0
            factor = int(np.clip(raw, min_factor, max_factor))
            pos_shuf = pos_df.sample(frac=factor, replace=True).reset_index(drop=True)
            batch_df = pd.concat([neg_shuf, pos_shuf]).sample(frac=1).reset_index(drop=True)
            for i in range(0, len(batch_df), batch_size):
                batch = batch_df.iloc[i:i+batch_size]
                Xb = np.stack(batch['X'].values).astype('float32')
                Xb_s = x_scaler.transform(Xb.reshape(-1, Xb.shape[-1])).reshape(Xb.shape)
                yb = batch['y'].values.reshape(-1,1).astype('float32')
                yb_s = y_scaler.transform(yb).ravel()
                spk = batch['spike'].values.astype('float32').reshape(-1,1)
                yield Xb_s, {'vol': yb_s, 'spike': spk}

    def val_generator():
        W, D = train_df.iloc[0]['X'].shape
        while True:
            for i in range(0, len(val_df), batch_size):
                batch = val_df.iloc[i:i+batch_size]
                Xb = np.stack(batch['X'].values).astype('float32')
                Xb_s = x_scaler.transform(Xb.reshape(-1, D)).reshape(-1, W, D)
                yb_s = y_scaler.transform(batch['y'].values.reshape(-1,1)).ravel()
                spk = batch['spike'].values.astype('float32').reshape(-1,1)
                yield Xb_s, {'vol': yb_s, 'spike': spk}

    steps  = len(train_df) // batch_size
    vsteps = int(np.ceil(len(val_df) / batch_size))
    W, D = train_df.iloc[0]['X'].shape

    inp    = Input(shape=(W, D), name="input_sequence")
    x      = Bidirectional(LSTM(128, return_sequences=True), name="bidi_lstm_1")(inp)
    x      = Dropout(0.2, name="dropout")(x)
    h      = Bidirectional(LSTM(128), name="bidi_lstm_2")(x)
    h      = LayerNormalization(name="layer_norm")(h)
    fusion = Dense(64, activation="relu", name="fusion")(h)

    spike_logits = Dense(1, name="spike_logits")(fusion)
    spike_scaled = Lambda(
        lambda t: t / temperature,
        output_shape=lambda s: s,
        name="spike_scaled"
    )(spike_logits)
    spike_pred   = Activation("sigmoid", name="spike")(spike_scaled)

    expert_norm  = Dense(1, name="expert_normal")(fusion)
    expert_spike = Dense(1, name="expert_spike")(fusion)

    one_minus_spike = Lambda(
        lambda t: 1 - t,
        output_shape=lambda s: s,
        name="one_minus_spike"
    )(spike_pred)
    norm_contrib    = Multiply(name="norm_contrib")([one_minus_spike, expert_norm])
    spike_contrib   = Multiply(name="spike_contrib")([spike_pred,    expert_spike])
    vol_pred        = Add(name="vol")([norm_contrib, spike_contrib])

    model = Model(inputs=inp, outputs=[vol_pred, spike_pred], name="moe_no_K")


    # --- Loss functions ---
    def weighted_mse(y_true, y_pred):
        abs_y = K.abs(y_true)
        w = 1.0 + abs_y / (K.mean(abs_y) + K.epsilon())
        return K.mean(w * K.square(y_true - y_pred))

    def log_cosh_loss(y_true, y_pred):
        def _log_cosh(x):
            return x + tf.math.softplus(-2.0 * x) - tf.math.log(2.0)
        return K.mean(_log_cosh(y_pred - y_true))

    def focal_bce(gamma=2.0, alpha=0.50):
        bce = tf.keras.losses.BinaryCrossentropy()
        def loss(y_true, y_pred):
            p_t = y_true * y_pred + (1-y_true)*(1-y_pred)
            return K.mean(alpha * K.pow(1-p_t, gamma) * bce(y_true, y_pred))
        return loss

    def zero_loss(y_true, y_pred):
        return K.zeros_like(K.mean(y_pred))

    # --- Callbacks ---
    warmup_cb = WarmUp(steps, learning_rate)
    es_cb = EarlyStopping(monitor='val_vol_loss', mode='min', patience=es_patience, restore_best_weights=True, verbose=0)
    rlr_cb = ReduceLROnPlateau(monitor='val_vol_loss', mode='min', factor=0.5, patience=lr_patience, min_lr=1e-7, verbose=0)

    # --- Stage 1: Weighted MSE only ---
    stage1_epochs = epochs // 2
    model.compile(
        optimizer=Adam(learning_rate),
        loss={'vol': weighted_mse, 'spike': zero_loss},
        loss_weights={'vol': 1.0, 'spike': 0.0},
        metrics={'vol': tf.keras.metrics.RootMeanSquaredError()}
    )
    history1 = model.fit(
        train_generator(),
        validation_data=val_generator(),
        steps_per_epoch=steps,
        validation_steps=vsteps,
        epochs=stage1_epochs,
        callbacks=[warmup_cb, es_cb, rlr_cb],
        verbose=0
    )

    # --- Stage 2: Combined vol loss + spike BCE ---
    stage2_epochs = epochs - stage1_epochs
    model.compile(
        optimizer=Adam(learning_rate),
        loss={'vol': log_cosh_loss, 'spike': focal_bce()},
        loss_weights={'vol': 1.0, 'spike': 0.5},
        metrics={'vol': tf.keras.metrics.RootMeanSquaredError(), 'spike': 'accuracy'}
    )
    history2 = model.fit(
        train_generator(),
        validation_data=val_generator(),
        steps_per_epoch=steps,
        validation_steps=vsteps,
        epochs=stage1_epochs + stage2_epochs,
        initial_epoch=stage1_epochs,
        callbacks=[warmup_cb, es_cb, rlr_cb],
        verbose=0
    )

    # --- Inference and results ---
    inf = Model(inp, [vol_pred, spike_pred, expert_norm, expert_spike])
    inf.set_weights(model.get_weights())
    flat_val = np.vstack(val_df['X'].values).astype('float32')
    Xv = x_scaler.transform(flat_val).reshape(-1, W, D)
    y_true = val_df['y'].values
    vols, probs, norms, spks, times = [], [], [], [], []
    for seq in Xv:
        t0 = time.perf_counter()
        v, p, n, s = inf.predict(seq[np.newaxis], verbose=0)
        times.append(time.perf_counter() - t0)
        vols.append(v.ravel()[0]); probs.append(p.ravel()[0])
        norms.append(n.ravel()[0]); spks.append(s.ravel()[0])

    results = val_df[['time_id','start_time']].copy()
    results['y_true'] = y_true
    results['y_pred'] = y_scaler.inverse_transform(np.array(vols).reshape(-1,1)).ravel()
    results['spike_prob'] = probs
    results['expert_normal'] = y_scaler.inverse_transform(np.array(norms).reshape(-1,1)).ravel()
    results['expert_spike'] = y_scaler.inverse_transform(np.array(spks).reshape(-1,1)).ravel()
    results['inference_time'] = times

    out_dir = os.path.join('models','lstm')
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, f"{model_name}.h5"), save_format='tf')
    with open(os.path.join(out_dir, f"{model_name}_scalers.pkl"), 'wb') as f:
        pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)

    tf.keras.backend.clear_session()
    del model, inf
    gc.collect()

    return history1, history2, results
