# ncf_model.py
# Neural Collaborative Filtering Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from sklearn.utils import shuffle
import pickle
import os

class NCFModel:
    """
    Neural Collaborative Filtering model combining GMF and MLP
    """

    def __init__(self, num_users, num_attractions, embedding_dim=50, mlp_layers=[64, 32, 16]):
        """
        Initialize the NCF model

        Args:
            num_users: Number of unique users
            num_attractions: Number of unique attractions
            embedding_dim: Dimension of embedding vectors
            mlp_layers: List of layer sizes for MLP
        """
        self.num_users = num_users
        self.num_attractions = num_attractions
        self.embedding_dim = embedding_dim
        self.mlp_layers = mlp_layers

        self.model = None
        self.user_embedding_gmf = None
        self.attraction_embedding_gmf = None
        self.user_embedding_mlp = None
        self.attraction_embedding_mlp = None

        self.history = None

    def build_model(self):
        """
        Build the NCF model architecture (GMF + MLP)

        Returns:
            Compiled Keras model
        """
        # Input layers
        user_input = layers.Input(shape=(1,), name='user_input')
        attraction_input = layers.Input(shape=(1,), name='attraction_input')

        # GMF branch
        self.user_embedding_gmf = layers.Embedding(
            input_dim=self.num_users,
            output_dim=self.embedding_dim,
            name='user_embedding_gmf'
        )

        self.attraction_embedding_gmf = layers.Embedding(
            input_dim=self.num_attractions,
            output_dim=self.embedding_dim,
            name='attraction_embedding_gmf'
        )

        user_latent_gmf = layers.Flatten()(self.user_embedding_gmf(user_input))
        attraction_latent_gmf = layers.Flatten()(self.attraction_embedding_gmf(attraction_input))
        gmf_vector = layers.Multiply()([user_latent_gmf, attraction_latent_gmf])

        # MLP branch
        self.user_embedding_mlp = layers.Embedding(
            input_dim=self.num_users,
            output_dim=self.embedding_dim,
            name='user_embedding_mlp'
        )

        self.attraction_embedding_mlp = layers.Embedding(
            input_dim=self.num_attractions,
            output_dim=self.embedding_dim,
            name='attraction_embedding_mlp'
        )

        user_latent_mlp = layers.Flatten()(self.user_embedding_mlp(user_input))
        attraction_latent_mlp = layers.Flatten()(self.attraction_embedding_mlp(attraction_input))
        mlp_vector = layers.Concatenate()([user_latent_mlp, attraction_latent_mlp])

        # MLP layers
        for idx, layer_size in enumerate(self.mlp_layers):
            mlp_vector = layers.Dense(layer_size, activation='relu',
                                     name=f'mlp_layer_{idx}')(mlp_vector)
            mlp_vector = layers.Dropout(0.2)(mlp_vector)

        # Concatenate GMF and MLP
        concat_vector = layers.Concatenate()([gmf_vector, mlp_vector])

        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='output')(concat_vector)

        # Create model
        self.model = Model(inputs=[user_input, attraction_input], outputs=output)

        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )

        print("NCF model built successfully!")
        return self.model

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=20, batch_size=256, validation_split=0.1):
        """
        Train the NCF model

        Args:
            X_train: Training features (user_id, attraction_id)
            y_train: Training labels (ratings)
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Split ratio for validation if no validation set provided
        """
        if self.model is None:
            self.build_model()

        # Prepare input data
        user_train = X_train[:, 0]
        attraction_train = X_train[:, 1]

        # Shuffle data
        user_train, attraction_train, y_train = shuffle(user_train, attraction_train, y_train)

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001
            )
        ]

        # Train model
        if X_val is not None and y_val is not None:
            user_val = X_val[:, 0]
            attraction_val = X_val[:, 1]

            self.history = self.model.fit(
                x=[user_train, attraction_train],
                y=y_train,
                validation_data=([user_val, attraction_val], y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                x=[user_train, attraction_train],
                y=y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

        print("Model training completed!")

        return self.history

    def predict(self, user_id, attraction_id):
        """
        Predict rating for a user-attraction pair

        Args:
            user_id: User ID
            attraction_id: Attraction ID

        Returns:
            Predicted rating (0-1 scale)
        """
        if self.model is None:
            raise ValueError("Model not built or loaded. Call build_model() or load_model() first.")

        user_array = np.array([user_id])
        attraction_array = np.array([attraction_id])

        prediction = self.model.predict([user_array, attraction_array], verbose=0)

        return float(prediction[0][0])

    def recommend_top_k(self, user_id, k=10, exclude_rated=True, data_loader=None):
        """
        Generate top-k recommendations for a user

        Args:
            user_id: User ID
            k: Number of recommendations
            exclude_rated: Whether to exclude already rated attractions
            data_loader: DataLoader instance for getting user history

        Returns:
            List of (attraction_id, score) tuples
        """
        if self.model is None:
            raise ValueError("Model not built or loaded. Call build_model() or load_model() first.")

        # Get attractions to predict
        attraction_ids = np.arange(self.num_attractions)

        if exclude_rated and data_loader is not None:
            # Get user's rated attractions
            user_history = data_loader.get_user_history(user_id)
            if len(user_history) > 0:
                rated_attractions = user_history['attraction_id'].values
                # Exclude rated attractions
                attraction_ids = np.setdiff1d(attraction_ids, rated_attractions)

        # If no attractions left, return empty list
        if len(attraction_ids) == 0:
            return []

        # Create arrays for prediction
        user_array = np.full(len(attraction_ids), user_id)

        # Predict ratings in batches to avoid memory issues
        batch_size = 1024
        predictions = []

        for i in range(0, len(attraction_ids), batch_size):
            batch_attractions = attraction_ids[i:i+batch_size]
            batch_users = user_array[i:i+batch_size]

            batch_predictions = self.model.predict(
                [batch_users, batch_attractions],
                batch_size=batch_size,
                verbose=0
            ).flatten()

            predictions.extend(batch_predictions)

        predictions = np.array(predictions)

        # Get top-k attractions
        if len(predictions) > 0:
            top_indices = np.argsort(predictions)[::-1][:k]
            top_attractions = attraction_ids[top_indices]
            top_scores = predictions[top_indices]

            # Return as list of tuples
            recommendations = list(zip(top_attractions, top_scores))
        else:
            recommendations = []

        return recommendations

    def get_user_embeddings(self):
        """
        Get user embedding matrix

        Returns:
            User embedding matrix
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")

        # Average of GMF and MLP embeddings
        gmf_embeddings = self.user_embedding_gmf.get_weights()[0]
        mlp_embeddings = self.user_embedding_mlp.get_weights()[0]

        return (gmf_embeddings + mlp_embeddings) / 2

    def get_attraction_embeddings(self):
        """
        Get attraction embedding matrix

        Returns:
            Attraction embedding matrix
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")

        # Average of GMF and MLP embeddings
        gmf_embeddings = self.attraction_embedding_gmf.get_weights()[0]
        mlp_embeddings = self.attraction_embedding_mlp.get_weights()[0]

        return (gmf_embeddings + mlp_embeddings) / 2

    def save_model(self, path):
        """
        Save the model and embeddings

        Args:
            path: Directory path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")

        os.makedirs(path, exist_ok=True)

        # Save Keras model
        model_path = os.path.join(path, 'ncf_model.keras')
        self.model.save(model_path)

        # Save model metadata
        metadata = {
            'num_users': self.num_users,
            'num_attractions': self.num_attractions,
            'embedding_dim': self.embedding_dim,
            'mlp_layers': self.mlp_layers
        }

        metadata_path = os.path.join(path, 'model_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Load a saved model

        Args:
            path: Directory path containing saved model
        """
        # Load metadata
        metadata_path = os.path.join(path, 'model_metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        self.num_users = metadata['num_users']
        self.num_attractions = metadata['num_attractions']
        self.embedding_dim = metadata['embedding_dim']
        self.mlp_layers = metadata['mlp_layers']

        # Load Keras model
        model_path = os.path.join(path, 'ncf_model.keras')
        self.model = keras.models.load_model(model_path)

        # Get embedding layers
        self.user_embedding_gmf = self.model.get_layer('user_embedding_gmf')
        self.attraction_embedding_gmf = self.model.get_layer('attraction_embedding_gmf')
        self.user_embedding_mlp = self.model.get_layer('user_embedding_mlp')
        self.attraction_embedding_mlp = self.model.get_layer('attraction_embedding_mlp')

        print(f"Model loaded from {path}")

        return self.model