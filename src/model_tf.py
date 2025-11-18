import tensorflow as tf

class NeuralNetworkTF:
    def __init__(self, input_dim=2, hidden_dim=5, output_dim=1, lr=0.001, seed=42):
        tf.random.set_seed(seed)
        # Sequential model with ReLU hidden and sigmoid output
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(output_dim, activation='sigmoid')
        ])
        # Adam optimizer, low learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X, Y, epochs=500, verbose_interval=50):
        history = self.model.fit(X, Y, epochs=epochs, verbose=0)
        for i in range(0, epochs, verbose_interval):
            print(f"Epoch {i} | Loss {history.history['loss'][i]:.4f}")
        return history.history['loss']

    def predict(self, X, threshold=0.5):
        y_hat = self.model.predict(X)
        return (y_hat >= threshold).astype(int)
