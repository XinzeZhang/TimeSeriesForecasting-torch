import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

class DeepState(tf.keras.models.Model):
    """
    DeepState 模型
    """
    def __init__(self, lstm_units, latent_dim):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = 1
        
        # 注意，文章中使用了多层的 LSTM 网络，为了简单起见，本 demo 只使用一层
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.dense_l_prior = tf.keras.layers.Dense(latent_dim)
        self.dense_P_prior = tf.keras.layers.Dense(latent_dim, activation='softplus')
        self.dense_F = tf.keras.layers.Dense(latent_dim * latent_dim)
        self.dense_H = tf.keras.layers.Dense(output_dim * latent_dim)
        self.dense_b = tf.keras.layers.Dense(output_dim)
        self.dense_w = tf.keras.layers.Dense(latent_dim, activation='softplus')
        self.dense_v = tf.keras.layers.Dense(output_dim, activation='softplus')
    
    def call(self, inputs, initial_state=None, prior=True):
        batch_size, time_steps, _ = inputs.shape
        
        outputs, state_h, state_c = self.lstm(inputs, initial_state=initial_state)
        state = [state_h, state_c]
            
        F = tf.reshape(self.dense_F(outputs), [batch_size, time_steps, self.latent_dim, self.latent_dim])
        H = tf.reshape(self.dense_H(outputs), [batch_size, time_steps, self.output_dim, self.latent_dim])
        
        b = tf.expand_dims(self.dense_b(outputs), -1)
        w = tf.expand_dims(self.dense_w(outputs), -1)
        v = tf.expand_dims(self.dense_v(outputs), -1)
        
        Q = tf.matmul(w, tf.transpose(w, [0, 1, 3, 2]))
        R = tf.matmul(v, tf.transpose(v, [0, 1, 3, 2]))
        
        params = [F, H, b, Q, R]
        
        if prior:
            l = tf.expand_dims(self.dense_l_prior(outputs[:, :1, :]), -1)
            P = tf.linalg.diag(self.dense_P_prior(outputs[:, :1, :]))
            params += [l, P]
        
        return [params, state]

def kalman_step(F, H, b, Q, R, l, P, z=None):
    """
    卡尔曼滤波的单步操作
    """
    sampling = z is None

    l = tf.matmul(F, l)
    P = tf.matmul(tf.matmul(F, P), tf.transpose(F, [0, 1, 3, 2])) + Q
    z_pred = tf.matmul(H, l) + b
    S = tf.matmul(tf.matmul(H, P), tf.transpose(H, [0, 1, 3, 2])) + R
    if sampling:
        z = tfp.distributions.Normal(z_pred, S).sample()
    else:
        log_prob = tfp.distributions.Normal(z_pred, S).log_prob(z)
    K = tf.matmul(tf.matmul(P, tf.transpose(H, [0, 1, 3, 2])), tf.linalg.inv(S))
    y = z - z_pred
    l = l + tf.matmul(K, y)
    P = P - tf.matmul(tf.matmul(K, H), P)
    
    if sampling:
        return [l, P, z]
    return [l, P, log_prob]

def kalman_filtering(F, H, b, Q, R, l, P, z=None):
    """
    卡尔曼滤波
    """
    time_steps = F.shape[1]
    if z is None:
        samples = []
        for t in range(time_steps):
            Ft = F[:, t:t+1, :, :]
            Ht = H[:, t:t+1, :, :]
            bt = b[:, t:t+1, :, :]
            Qt = Q[:, t:t+1, :, :]
            Rt = R[:, t:t+1, :, :]
            l, P, zt = kalman_step(Ft, Ht, bt, Qt, Rt, l, P)
            samples.append(zt)
        return samples
    else:
        log_probs = []
        for t in range(time_steps):
            Ft = F[:, t:t+1, :, :]
            Ht = H[:, t:t+1, :, :]
            bt = b[:, t:t+1, :, :]
            Qt = Q[:, t:t+1, :, :]
            Rt = R[:, t:t+1, :, :]
            zt = z[:, t:t+1, :, :]
            l, P, log_prob = kalman_step(Ft, Ht, bt, Qt, Rt, l, P, zt)
            log_probs.append(log_prob)
        loss = -tf.reduce_sum(log_probs)
        return l, P, loss

LSTM_UNITS = 16
LATENT_DIM = 10
EPOCHS = 10

# 实例化模型
model = DeepState(LSTM_UNITS, LATENT_DIM)

# 指定优化器
optimizer = tf.keras.optimizers.Adam()

# 定义训练步
def train_step(x, z):
    with tf.GradientTape() as tape:
        params, _ = model(x)
        _, _, loss_value = kalman_filtering(*params, z)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_value.numpy()

# 数据处理（略）
# train_data = do_something()

# 训练
for epoch in range(EPOCHS):
    loss = []
    for x, z in train_data:
        loss.append(train_step(x, z))
    print('Epoch %d, Loss %.4f' % (epoch + 1, np.mean(loss))