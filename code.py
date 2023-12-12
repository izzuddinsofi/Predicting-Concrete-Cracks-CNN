"""
Predicting Concretet with Cracks in Images
"""

#1. Import the necessary packages and dataset
import matplotlib.pyplot as plt
import numpy as np
import os, datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications, layers
import pathlib as path

data_path = r"C:\Users\muhdi\Documents\Deep Learning\Cracked-Concrete-Project\data"
data_dir = path.Path(data_path)

#%%

#2. Data preparation
#Split into train-validation set
SEED = 12345
IMG_SIZE = (160,160)
BATCH_SIZE = 16

train_dataset = keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.3, subset='training', seed=SEED, shuffle=True,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE)
val_dataset = keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.3, subset='validation', seed=SEED, shuffle=True,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE)

#%%
#Further split validation dataset into validation-test split
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

# Create prefetch dataset for all 3 splits
AUTOTUNE = tf.data.AUTOTUNE
pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_validation = validation_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)

#Data preparation is completed at this step
#we have abundant data, so no data augmentation is needed

#%%
#3. We are applying transfer learning to create this model using MobileNetV2
#Create a layer for that preprocesses the input for the transfer learning model
preprocess_input = applications.mobilenet_v2.preprocess_input

#Create the base model by using MobileNetV2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

#Freeze the base model and view the base model structure
base_model.trainable = False
base_model.summary()

#%%
#3.1 Create classification layers with global average pooling and a dense layer
class_names = train_dataset.class_names
nClass = len(class_names)

global_avg_pooling = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(nClass, activation='softmax')

#3.2 Use functional API to construct the entire model
inputs = keras.Input(shape=IMG_SHAPE)
x = preprocess_input(inputs)
x = base_model(x)
x = global_avg_pooling(x)
outputs = output_layer(x)

model = keras.Model(inputs, outputs)

#view the model structure after adding the classification layer
model.summary()
#to view a representation of your model structure
tf.keras.utils.plot_model(model, show_shapes=True)

#%%
#3.3 Compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
loss = keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

#%%
#4 Perform model training
EPOCHS = 10
base_log_path = r"C:\Users\muhdi\Documents\Deep Learning\Cracked-Concrete-Project\log"
log_path= os.path.join(base_log_path, datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '__Cracked_Concrete_Project')
tb = keras.callbacks.TensorBoard(log_dir=log_path)
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)

history = model.fit(pf_train, validation_data=pf_validation, epochs=EPOCHS, callbacks=[es,tb])

#%%
#to view graph of train against validation, run the code below without the # key in prompt
#tensorboard --logdir "C:\Users\muhdi\Documents\Deep Learning\Cracked-Concrete-Project\log"

#%%
#5. Evaluate the trained model with test dataset
test_loss, test_accuracy = model.evaluate(pf_test)
print('______________________________Test Result______________________________')
print(f'Loss = {test_loss}')
print(f'Accuracy = {test_accuracy}')

#%%
#6. Deploy the model to make prediction as we know we have a good model
image_batch, label_batch = pf_test.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)
class_predictions = np.argmax(predictions, axis=1)

#%%
#7. Show some prediction results
plt.figure(figsize=(10,10))

for i in range(4):
    axs = plt.subplot(2,2,i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    current_prediction = class_names[class_predictions[i]]
    current_label = class_names[label_batch[i]]
    plt.title(f"Prediction: {current_prediction}, Actual: {current_label}")
    plt.axis('off')
    
save_path = r"C:\Users\muhdi\Documents\Deep Learning\Cracked-Concrete-Project\img"
plt.savefig(os.path.join(save_path,"result.png"),bbox_inches='tight')
plt.show()