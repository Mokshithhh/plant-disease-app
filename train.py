import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(
    "dataset",
    target_size=IMG_SIZE,
    subset="training"
)

val = datagen.flow_from_directory(
    "dataset",
    target_size=IMG_SIZE,
    subset="validation"
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output = tf.keras.layers.Dense(train.num_classes, activation='softmax')(x)

model = tf.keras.Model(base_model.input, output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train, validation_data=val, epochs=5)

model.save("model.h5")
print(train.class_indices)