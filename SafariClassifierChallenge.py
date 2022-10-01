import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


train_data_folder = 'safari\\training'
test_data_folder='safari\\test'
classnames = os.listdir(train_data_folder)
classnames.sort()
print(len(classnames), 'classes:')
print(classnames)

fig = plt.figure(figsize=(8, 12))
i = 0
for sub_dir in os.listdir(train_data_folder):
    i+=1
    img_file = os.listdir(os.path.join(train_data_folder,sub_dir))[0]
    img_path = os.path.join(train_data_folder, sub_dir, img_file)
    img = mpimg.imread(img_path)
    a=fig.add_subplot(1, len(classnames),i)
    a.axis('off')
    imgplot = plt.imshow(img)
    a.set_title(img_file)
plt.show()


from keras.preprocessing.image import ImageDataGenerator

img_size = (200, 200)
batch_size = 30


print("Getting Data...")
 # normalize pixel values
datagen = ImageDataGenerator(rescale=1./255)

print("Preparing training dataset...")
train_generator = datagen.flow_from_directory(
    train_data_folder,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

print("Preparing validation dataset...")
validation_generator = datagen.flow_from_directory(
    test_data_folder,
    target_size=img_size,
    batch_size=1,
    class_mode='categorical',
    subset='validation') # set as validation data
print(validation_generator.num_classes)

#classnames = list(train_generator.class_indices.keys())
print('Data generators ready',classnames)

filter_Size=32
filters_no=(6, 6)

# Define a CNN classifier network
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Define the model as a sequence of layers
model = Sequential()

# The input layer accepts an image and applies a convolution that uses 32 6x6 filters and a rectified linear unit activation function
model.add(Conv2D(filter_Size, filters_no, input_shape=train_generator.image_shape, activation='relu'))

# Next we'll add a max pooling layer with a 2x2 patch
model.add(MaxPooling2D(pool_size=(2,2)))

# We can add as many layers as we think necessary - here we'll add another convolution and max pooling layer
model.add(Conv2D(filter_Size, filters_no, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# And another set
model.add(Conv2D(filter_Size, filters_no, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# A dropout layer randomly drops some nodes to reduce inter-dependencies (which can cause over-fitting)
model.add(Dropout(0.2))

# Flatten the feature maps 
model.add(Flatten())

# Generate a fully-connected output layer with a predicted probability for each class
# (softmax ensures all probabilities sum to 1)
model.add(Dense(train_generator.num_classes, activation='softmax'))

# With the layers defined, we can now compile the model for categorical (multi-class) classification
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())



# Train the model over 5 epochs using 30-image batches and using the validation holdout dataset for validation
num_epochs = 5
history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    #validation_steps = validation_generator.samples // batch_size,
    epochs = num_epochs)




from matplotlib import pyplot as plt

epoch_nums = range(1,num_epochs+1)
training_loss = history.history["loss"]
accuracy = history.history["accuracy"]
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, accuracy)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()



# Tensorflow doesn't have a built-in confusion matrix metric, so we'll use SciKit-Learn
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Save the trained model
modelFileName = 'models/safari_classifier.h5'
model.save(modelFileName)
del model  # deletes the existing model variable
print('model saved as', modelFileName)




from keras import models
import numpy as np
from random import randint
import os


# Use the classifier to predict the class
model = models.load_model(modelFileName) # loads the saved model




print("Generating predictions from validation data...")
# Get the image and label arrays for the first batch of validation data
def predict_image(classifier, image):
    from tensorflow import convert_to_tensor
    # The model expects a batch of images as input, so we'll create an array of 1 image
    print(image)
    print(image.shape)
    imgfeatures = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

    # We need to format the input to match the training data
    # The generator loaded the values as floating point numbers
    # and normalized the pixel values, so...
    imgfeatures = imgfeatures.astype('float32')
    imgfeatures /= 255
    
    # Use the model to predict the image class
    class_probabilities = classifier.predict(imgfeatures)
    
    # Find the class predictions with the highest predicted probability
    index = int(np.argmax(class_probabilities, axis=1)[0])
    return index


fig = plt.figure(figsize=(8, 12))
i = 0
X_test=[]
y_test=[]
for img_file in os.listdir(test_data_folder):
    i+=1
    img_path = os.path.join(test_data_folder, img_file)
    x_test = mpimg.imread(img_path)
    print(x_test.shape)
    #print(x_test.resize(128,128,3))
    # Get the image class prediction
    X_test.append(np.array(x_test))
    y_test.append(img_file)
    # Get the image class prediction
    prediction = predict_image(model,np.array(x_test))
    print(prediction)
    a=fig.add_subplot(1, len(classnames),i)
    a.axis('off')
    imgplot = plt.imshow(x_test)
    a.set_title(classnames[prediction])
plt.show()

imgfeatures=np.array(X_test)
imgfeatures = imgfeatures.astype('float32')
imgfeatures /= 255
    
# Use the model to predict the image class
class_probabilities = model.predict(imgfeatures)


# The model returns a probability value for each class
# The one with the highest probability is the predicted class
predictions = np.argmax(class_probabilities, axis=1)

print(predictions)