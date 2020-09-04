import tensorflow as tf
import numpy as np
import getConfig
gConfig={}
gConfig=getConfig.get_config(config_file='config.ini.all2')


def layernormalization(inputs,epsilon=1e-6):

    inputsShape = inputs.get_shape() #[batch_size, sequence_length, embedding_size]

    paramsShape = inputsShape[-1:]

    mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)

    beta = tf.Variable(tf.zeros(paramsShape))

    gamma = tf.Variable(tf.ones(paramsShape))

    normalized = (inputs - mean) / ((variance + epsilon) ** .5)

    outputs = gamma * normalized + beta
    return outputs

def get_angles(pos, i, d_model):
    angle_rates = 1/np.power(10000,(2*(i//2))/np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):

    angle_rads = get_angles(np.arange(position)[:, np.newaxis],np.arange(d_model)[np.newaxis,:],d_model)

    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:,1::2])

    pos_encoding = np.concatenate([sines,cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis,...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q, k, v, mask):


    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model,  dff, num_heads, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, dff, num_heads,  input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, dff, num_heads, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, dff, num_heads, input_vocab_size, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, dff, num_heads,
                               input_vocab_size, rate)

        self.ffn_out = tf.keras.layers.Dense(2,activation='softmax')
        self.dropout1 = tf.keras.layers.Dropout(rate)


    def call(self, inp, training, enc_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        out_shape = gConfig['sentence_size']*gConfig['embedding_size']
        enc_output = tf.reshape(enc_output,(-1,out_shape))
        ffn = self.dropout1(enc_output,training=training)
        ffn_out = self.ffn_out(ffn)

        return ffn_out

class TransformerMulti(tf.keras.Model):
    def __init__(self, num_feat1, num_feat2, num_layers, d_model, dff, num_heads, input_vocab_size, rate=0.1):
        super(TransformerMulti, self).__init__()

        self.encoder = Encoder(num_layers, d_model, dff, num_heads,
                               input_vocab_size, rate)

        self.ffn_out = tf.keras.layers.Dense(2,activation='softmax')
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.embedding1 = tf.keras.layers.Embedding(num_feat1, d_model)
        self.embedding2 = tf.keras.layers.Embedding(num_feat2, d_model)
        self.enc_layers = [EncoderLayer(d_model, dff, num_heads, rate)
                           for _ in range(num_layers)]
        for i in range(num_layers):
            setattr(self, 'dense_' + str(i), tf.keras.layers.Dense(d_model))
            setattr(self, 'batchNorm_' + str(i), tf.keras.layers.BatchNormalization())
            setattr(self, 'activation_' + str(i), tf.keras.layers.Activation('relu'))
            setattr(self, 'dropout_' + str(i), tf.keras.layers.Dropout(rate))

    def call(self, inp1, inp2, inp3, training, enc_padding_mask):
        enc_output = self.encoder(inp3, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        feat1_index = tf.cast(np.array([range(gConfig['num_feat1'])]*int(tf.shape(inp1)[0])), tf.float32)
        feat2_index = tf.cast(np.array([range(gConfig['num_feat2'])]*int(tf.shape(inp2)[0])), tf.float32)
        inp1_emb = self.embedding1(feat1_index) # (batch_size, num_feat, d_model)
        inp2_emb = self.embedding2(feat2_index) # (batch_size, num_feat, d_model)

        inp1 = tf.expand_dims(inp1, axis=-1)
        inp2 = tf.expand_dims(inp2, axis=-1)
        #print(inp1,inp1_emb,inp2,inp2_emb)
        inp1_emb = tf.math.multiply(inp1_emb, inp1)
        inp2_emb = tf.math.multiply(inp2_emb, inp2)

        for i in range(gConfig['num_layers']):
            inp2_emb = self.enc_layers[i](inp2_emb, training, None)# (batch_size, num_feat, d_model)
        for i in range(gConfig['num_layers']):
            inp1_emb = getattr(self, 'dense_' + str(i))(inp1_emb)
            inp1_emb = getattr(self, 'batchNorm_' + str(i))(inp1_emb)
            inp1_emb = getattr(self, 'activation_' + str(i))(inp1_emb)
            inp1_emb = getattr(self, 'dropout_' + str(i))(inp1_emb)
        out_shape = gConfig['sentence_size']*gConfig['embedding_size']
        enc_output = tf.reshape(enc_output,(-1,out_shape))
        inp1_output = tf.reshape(inp1_emb,(-1,gConfig['num_feat1'] * gConfig['embedding_size']))
        inp2_output = tf.reshape(inp2_emb,(-1,gConfig['num_feat2'] * gConfig['embedding_size']))
        concat_input = tf.concat((enc_output, inp1_output, inp2_output), axis=1)
        ffn = self.dropout1(concat_input,training=training)
        ffn_out = self.ffn_out(ffn)

        return ffn_out

def evaluate(inp1, inp2, inp3,tar):
    inp2 = tf.cast(inp2, dtype=tf.float32)
    inp1 = tf.cast(inp1, dtype=tf.float32)
    enc_padding_mask = create_padding_mask(inp3)
    predictions = transformerMulti(inp1, inp2, inp3,False,enc_padding_mask)
    predictions = predictions[:,-1]
    loss = tf.losses.binary_crossentropy(tar,predictions)
    return train_loss(loss),train_accuracy(tar,predictions)

def prediction(inp1, inp2, inp3, tar):
    inp2 = tf.cast(inp2, dtype=tf.float32)
    inp1 = tf.cast(inp1, dtype=tf.float32)
    enc_padding_mask = create_padding_mask(inp3)
    predictions = transformerMulti(inp1, inp2, inp3,False,enc_padding_mask)
    predictions = predictions[:,-1]
    predictions = tf.reshape(predictions,(-1,1))
    tar = tf.reshape(tar,(-1,1))
    #predictions = tf.reshape(predictions,(1,gConfig['batch_size']))
    data = np.concatenate((tar,predictions),axis=1)
    return data

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(gConfig['embedding_size'])

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.AUC(name='train_auc')

transformerMulti = TransformerMulti(gConfig['num_feat1'],gConfig['num_feat2'],gConfig['num_layers'], gConfig['embedding_size'], gConfig['diff'], gConfig['num_heads'],
                          gConfig['vocabulary_size'], gConfig['drop_rate'])
ckpt = tf.train.Checkpoint(transformerMulti=transformerMulti,optimizer=optimizer)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq,0),tf.float32)
    return seq[:,tf.newaxis,tf.newaxis,:]

def step(inp1, inp2, inp3,tar,train_status=True):
    enc_padding_mask = create_padding_mask(inp3)
    inp2 = tf.cast(inp2, dtype=tf.float32)
    inp1 = tf.cast(inp1, dtype=tf.float32)
    if train_status:
        with tf.GradientTape() as tape:
            predictions = transformerMulti(inp1, inp2, inp3, True, enc_padding_mask)
            tar = tf.keras.utils.to_categorical(tar,2)
            loss = tf.losses.categorical_crossentropy(tar,predictions)
    #loss = tf.losses.binary_crossentropy(tar,predictions)
        #print(loss)

    gradients = tape.gradient(loss, transformerMulti.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformerMulti.trainable_variables))

    return train_loss(loss), train_accuracy(tar, predictions)

