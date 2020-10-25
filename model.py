import tensorflow as tf
import numpy as np
from tokenizers import Tokenizer

def main(load=True):
    tokenizer = Tokenizer.from_file("bpe-fi.tokenizer.json")
    vocab = tokenizer.get_vocab()

    VOCAB_SIZE = max(vocab.values()) + 1
    INPUT_LEN = 5
    DIM = 128
    BATCH_SIZE = 2 ** 10

    if not load:
        inputs = tf.keras.Input(shape=(None,), name="title")

        # Embed each word in the title into a 64-dimensional vector
        x0 = tf.keras.layers.Embedding(VOCAB_SIZE, DIM)(inputs)
        x1 = tf.keras.layers.LSTM(DIM)(x0)
        x2 = tf.keras.layers.Dense(VOCAB_SIZE, name="priority")(x1)

        model = tf.keras.Model(inputs=inputs, outputs=x2)
    else:
        model = tf.keras.models.load_model('./saved_model')

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        #loss_weights=[1.0, 0.2],
    )

    data = np.load("out.npy")#[00000]
    dlen = len(data)

    train_x = [ data[i : i + INPUT_LEN] for i in range(dlen - INPUT_LEN - 1)] 
    train_y = [ data[i + INPUT_LEN] for i in range(dlen - INPUT_LEN - 1)] 

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_y = tf.keras.utils.to_categorical(train_y, num_classes=VOCAB_SIZE)

    print(train_x.shape, train_y.shape)

    model.fit(train_x, train_y, epochs=3, batch_size=BATCH_SIZE)

    val_data = np.random.randint(VOCAB_SIZE, size=(1, INPUT_LEN))

    y_log = model(val_data)
    y = tf.nn.softmax(y_log)
    print(y)

    model.save('./saved_model')

if __name__ == '__main__':
    main()