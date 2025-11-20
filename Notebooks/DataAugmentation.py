from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

root_data_dir = r"D:\Projects\MulticlassFishImageClassification\Project_data\data"

train_dir = os.path.join(root_data_dir, 'train')
valid_dir = os.path.join(root_data_dir, 'val')
test_dir = os.path.join(root_data_dir, 'test')

# Image settings
IMG_SIZE = (256, 256)
BATCH_SIZE = 32

def data_loader(train_dir,valid_dir,test_dir):
    ## Data Augmentation applied only for Training data

    train_datagen = ImageDataGenerator(
        rescale=1./255,              
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        shear_range=0.1,
        horizontal_flip=True,
        vertical_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,           # Path to train folder
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        color_mode="rgb"
    )

    #  Appy Only rescaling on validation data

    valid_datagen = ImageDataGenerator(rescale=1./255)

    valid_generator = valid_datagen.flow_from_directory(
        directory= valid_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        color_mode="rgb"
    )

    # Appy Only rescaling on validation data

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        color_mode="rgb"
    )

    return train_generator, valid_generator,test_generator

load_data = data_loader(train_dir,valid_dir,test_dir)


