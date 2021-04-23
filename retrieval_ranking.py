import os
import pprint
import tempfile


from typing import Dict, Text

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# Get data
# ratings = tfds.load('movielens/100k-ratings', split="train")
# movies = tfds.load('movielens/100k-movies', split="train")

path = 'data\\mockdata.csv'
df = pd.read_csv(path)
df.head()
print(df)

mockdata = tf.data.Dataset.from_tensor_slices((tf.cast(df['user_id'].values, tf.int64),
                                               tf.cast(
    df['campsite_id'].values, tf.int64),
    tf.cast(df['user_rating'].values, tf.float32)))

mockdata = tf.data.Dataset.from_tensor_slices(dict(df))

print('mockdata:')
print(list(mockdata.as_numpy_iterator()))
print(mockdata)

# Select the basic features.
ratings = mockdata.map(lambda x: {
    "campsite_id": x['campsite_id'],
    "user_id": x['user_id'],
    "user_rating": x['user_rating'],
})
campings = mockdata.map(lambda x: x['campsite_id'])

print(ratings)
print(campings)

# Randomly shuffle data and split between train and test.
tf.random.set_seed(42)
shuffled = ratings.shuffle(100, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80)
test = shuffled.skip(80).take(20)

# Define vocabularies
camping_ids = campings.batch(1000)
user_ids = ratings.batch(1000).map(lambda x: x["user_id"])

unique_campsites = np.unique(np.concatenate(list(camping_ids)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

print(unique_user_ids)
print(unique_campsites)


# Combine to create model
class MovielensModel(tfrs.models.Model):

    def __init__(self, rating_weight: float, retrieval_weight: float) -> None:
        # We take the loss weights in the constructor: this allows us to instantiate
        # several model objects with different loss weights.

        super().__init__()

        embedding_dimension = 32

        # User and campsite models.
        self.campsite_model: tf.keras.layers.Layer = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.IntegerLookup(
                vocabulary=unique_campsites, mask_token=None),
            tf.keras.layers.Embedding(
                len(unique_campsites) + 1, embedding_dimension)
        ])
        self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.IntegerLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(
                len(unique_user_ids) + 1, embedding_dimension)
        ])

        # A small model to take in user and movie embeddings and predict ratings.
        # We can make this as complicated as we want as long as we output a scalar
        # as our prediction. These are the layers the DNN uses to predict.
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

        # The tasks.
        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=campings.batch(128).map(self.campsite_model)
            )
        )

        # The loss weights.
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

    def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the movie features and pass them into the movie model.
        camping_embeddings = self.campsite_model(features["campsite_id"])

        return (
            user_embeddings,
            camping_embeddings,
            # We apply the multi-layered rating model to a concatentation of
            # user and movie embeddings.
            self.rating_model(
                tf.concat([user_embeddings, camping_embeddings], axis=1)
            ),
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

        ratings = features.pop("user_rating")

        user_embeddings, camping_embeddings, rating_predictions = self(
            features)

        # We compute the loss for each task.
        rating_loss = self.rating_task(
            labels=ratings,
            predictions=rating_predictions,
        )
        retrieval_loss = self.retrieval_task(
            user_embeddings, camping_embeddings)

        # And combine them using the loss weights.
        return (self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss)


model = MovielensModel(rating_weight=1.0, retrieval_weight=1.0)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

cached_train = train.shuffle(128).batch(128).cache()
cached_test = test.batch(13).cache()

model.fit(cached_train, epochs=3)
metrics = model.evaluate(cached_test, return_dict=True)

print(
    f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")

# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
# recommends movies out of the entire movies dataset.
index.index(campings.batch(10).map(model.campsite_model), campings)

_, titles = index(tf.constant([7]))
print(f"Recommendations for user 7: {titles[0]}")
