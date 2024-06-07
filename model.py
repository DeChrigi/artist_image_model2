import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping

#Datenset bauen
def buildDataset():
    # Pfad zu den Daten
    dataset_dir = './dataset/train_images'

    # ImageDataGenerator für die Datenvorbereitung with Image Augmentation
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=0.2,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # Trainings- und Validierungsdaten
    train_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=(150, 150),
        batch_size=36,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=(150, 150),
        batch_size=34,
        class_mode='categorical',
        subset='validation'
    )
    # Konvertiere zu tf.data.Dataset
    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 150, 150, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, len(train_generator.class_indices)), dtype=tf.float32)
        )
    )
    validation_dataset = tf.data.Dataset.from_generator(
        lambda: validation_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 150, 150, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, len(validation_generator.class_indices)), dtype=tf.float32)
        )
    )

    # Wiederhole die Datensätze
    train_dataset = train_dataset.repeat()
    validation_dataset = validation_dataset.repeat()
    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = validation_generator.samples // validation_generator.batch_size
    num_of_classes = len(train_generator.class_indices)

    print(f"Steps per epoch training: {steps_per_epoch}")
    print(f"Steps per epoch validation: {validation_steps}")
    print(f"Number of training samples: {train_generator.samples}")
    print(f"Number of validation samples: {validation_generator.samples}")
    print(f"Number of classes: {num_of_classes}")

    return train_dataset, validation_dataset, steps_per_epoch, validation_steps, num_of_classes

# Speichern des Modells
def saveModel(model, name):
    # Speichern des Modells im HDF5-Format
    pathname = f"./trained_models/{name}"

    model.save(pathname)


def buildModelV2():

    train_dataset, validation_dataset, steps_per_epoch, validation_steps, num_of_classes = buildDataset()
    # Modellarchitektur
    model = Sequential([
        tf.keras.layers.Input(shape=(150, 150, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_of_classes, activation='softmax')
    ])

    # Kompilieren des Modells
    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Modelltraining
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps,
        epochs=25
    )

    # Modellbewertung
    loss, accuracy = model.evaluate(validation_dataset, steps=validation_steps)
    print(f'Validation loss: {loss}')
    print(f'Validation accuracy: {accuracy}')

    saveModel(model, 'my_model_v2.h5')

def buildModelV4():

    train_dataset, validation_dataset, steps_per_epoch, validation_steps, num_of_classes = buildDataset()
    # Modellarchitektur
    model = Sequential([
        tf.keras.layers.Input(shape=(150, 150, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_of_classes, activation='softmax')
    ])
    # Kompilieren des Modells
    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # Modelltraining
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps,
        epochs=25
    )

    # Modellbewertung
    loss, accuracy = model.evaluate(validation_dataset, steps=validation_steps)
    print(f'Validation loss: {loss}')
    print(f'Validation accuracy: {accuracy}')
    
    # Speichern des Modells im HDF5-Format
    saveModel(model, 'my_model_v4.h5')

def buildModelV5():

    train_dataset, validation_dataset, steps_per_epoch, validation_steps, num_of_classes = buildDataset()
    # Modellarchitektur
    model = Sequential([
        tf.keras.layers.Input(shape=(150, 150, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_of_classes, activation='softmax')
    ])
    # Kompilieren des Modells
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # Modelltraining
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps,
        epochs=25
    )
    
    # Modellbewertung
    loss, accuracy = model.evaluate(validation_dataset, steps=validation_steps)
    print(f'Validation loss: {loss}')
    print(f'Validation accuracy: {accuracy}')

    # Speichern des Modells im HDF5-Format
    saveModel(model, 'my_model_v5.h5')

def buildModelV6():
    
    train_dataset, validation_dataset, steps_per_epoch, validation_steps, num_of_classes = buildDataset()
    # Modellarchitektur
    base_model = VGG16(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model
    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_of_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # EarlyStopping Callback definieren
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps,
        epochs=25,
        callbacks=[early_stopping]
    )

    # Modellbewertung
    loss, accuracy = model.evaluate(validation_dataset, steps=validation_steps)
    print(f'Validation loss: {loss}')
    print(f'Validation accuracy: {accuracy}')

    saveModel(model, 'my_model_v6.h5')