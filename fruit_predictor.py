# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 16:30:56 2024

@author: Jungyu Lee

"""
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

batch_size = 32
img_height = 32
img_width = 32

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'Fruit-Images-Dataset/',
    seed=21,
    image_size=(img_height, img_width),
    batch_size=batch_size)


total_size = len(dataset)
train_size = int(0.6 * total_size)  
val_size = int(0.2 * total_size)    
test_size = total_size - train_size - val_size  

train_data = dataset.take(train_size)
test_data = dataset.skip(train_size)
val_data = test_data.skip(test_size)
test_data = test_data.take(test_size)

normalize = layers.experimental.preprocessing.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalize(x), y))
val_data = val_data.map(lambda x, y: (normalize(x), y))
test_data = test_data.map(lambda x, y: (normalize(x), y))

model = models.Sequential([
    layers.Flatten(input_shape=(img_height, img_width, 3)),
    layers.Dense(784, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(4)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_data, validation_data=val_data, epochs=30)

test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test loss: {test_loss:.2f}")
print(f"Test accuracy: {test_accuracy:.2f}")

class_names = ['Apple', 'Banana', 'Beetroot', 'Cactus fruit']

for images, labels in test_data.take(1):
    predictions = model.predict(images)
    predictions_labels = np.argmax(predictions, axis=1)
    
    plt.figure(figsize=(15, 15))

    for i in range(batch_size):
        plt.subplot(5, 7, i + 1)  
        display_image = (images[i].numpy() * 255).astype("uint8")
        plt.imshow(display_image)
        
        actual_label = class_names[labels[i]]
        predicted_label = class_names[predictions_labels[i]]
        
        plt.title(f"Actual: {actual_label}\nPredicted: {predicted_label}", fontsize=10)
        plt.axis("off")

    plt.subplots_adjust(hspace=0.5)  
    plt.tight_layout()
    plt.show()
