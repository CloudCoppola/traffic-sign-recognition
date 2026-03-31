import pandas as pd
import tensorflow as tf

def load_labels(label_path: str):
    return pd.read_csv(label_path)

def build_datasets(dataset_path: str, img_size=(224, 224), batch_size=32, seed=42):

    train_ds = tf.keras.preprocessing.image_dataset_directory(
        dataset_path,
        validation_split=0.2,
        subset='training',
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    vals_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset='validation',
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    return train_ds, vals_ds

def map_class_names(train_ds, label_df):
    class_names = []
    for i in train_ds.class_names:
        class_names.append(label_df['Name'][int(i)])
    return class_names
