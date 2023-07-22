import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.utils import img_to_array, load_img
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight
from tqdm.notebook import tqdm as tqdm
import os
import cv2
import seaborn as sns


def load_data():
    Train_img = r"D:\Materials\Graduation Project\Dataset\aptos2019\train_images\train_images"
    Train_csv = r"D:\Materials\Graduation Project\Dataset\aptos2019\train_1.csv"
    Test_img = r"/D:\Materials\Graduation Project\Dataset\idridtest\TestDR\TestDR"
    Test_csv = r"D:\Materials\Graduation Project\Dataset\idridtest\IDRiD_Testing Labels.csv"
    traindf = pd.read_csv(Train_csv, dtype=str)
    testdf = pd.read_csv(Test_csv, dtype=str)
    traindf['diagnosis'] = traindf['diagnosis'].astype(str)
    traindf['id_code'] = traindf['id_code'].astype(str) + '.png'
    testdf['Image name'] = testdf['Image name'].astype(str) + '.jpg'
    return Train_img, Train_csv, Test_img, Test_csv, traindf, testdf

def calculate_class_weights(traindf):
    counts = traindf['diagnosis'].value_counts()
    print(counts)
    max_count = counts.max()
    normalized_counts = []
    for count in counts:
        normalized_counts.append(max_count / count)
    print(normalized_counts)
    class_weights = {0: normalized_counts[0], 1: normalized_counts[2], 2: normalized_counts[1], 3: normalized_counts[4], 4: normalized_counts[3]}
    return class_weights

output_dir = 'D:\Materials\Graduation Project\Dataset\TrainAPTOSIMAGEPROCESSING'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def preprocessing(files):
    img_list = []
    for i in tqdm(files):
        image = cv2.imread(Train_img + "/" + i)
        image = cv2.resize(image, (240, 240))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        copy = image.copy()
        copy = cv2.cvtColor(copy, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(copy, (5, 5), 0)

        thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)[1]

        contour = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contour[0][0]
        contour = contour[:, 0, :]

        x1 = tuple(contour[contour[:, 0].argmin()])[0]
        y1 = tuple(contour[contour[:, 1].argmin()])[1]
        x2 = tuple(contour[contour[:, 0].argmax()])[0]
        y2 = tuple(contour[contour[:, 1].argmax()])[1]

        x = int(x2 - x1) * 4 // 50
        y = int(y2 - y1) * 5 // 50

        copy2 = image.copy()
        if x2 - x1 > 100 and y2 - y1 > 100:
            copy2 = copy2[y1 + y: y2 - y, x1 + x: x2 - x]
            copy2 = cv2.resize(copy2, (240, 240))

        lab = cv2.cvtColor(copy2, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=((8, 8)))
        cl = clahe.apply(l)

        merge = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(merge, cv2.COLOR_LAB2RGB)
        med_blur = cv2.medianBlur(final_img, 3)
        back_gorund = cv2.medianBlur(final_img, 37)

        mask = cv2.addWeighted(med_blur, 1, back_gorund, -1, 255)
        final = cv2.bitwise_and(mask, med_blur)
        img_list.append(final)
        output_filename = os.path.join(output_dir, i)
        final= cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_filename, final)
    return img_list

def create_data_generators(output_dir, traindf, Test_img, testdf):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       validation_split=0.2,
                                       horizontal_flip=True,
                                       )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=output_dir,
        x_col="id_code",
        y_col="diagnosis",
        batch_size=32,
        class_mode="categorical",
        target_size=(224, 224),
        shuffle=True,
        subset='training')

    valid_generator = train_datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=output_dir,
        x_col="id_code",
        y_col="diagnosis",
        batch_size=32,
        class_mode="categorical",
        target_size=(224, 224),
        shuffle=True,
        subset='validation')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=testdf,
        directory=Test_img,
        x_col="Image name",
        y_col="Retinopathy grade",
        target_size=(224, 224),
        batch_size=16,
        shuffle=False,
        class_mode="categorical")

    return train_generator, valid_generator, test_generator

def build_model(train_generator,valid_generator,traindf,weights):
    base_model = tf.keras.applications.DenseNet121(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(256, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(5, activation='softmax'))

    model.summary()

    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]

    lrd = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.8, min_lr=1e-6)

    mcp = ModelCheckpoint('D:\Materials\Graduation Project\Models\DenseNet121.h5')

    es = EarlyStopping(verbose=1, patience=2)
    model.compile(optimizer='Adam', loss="categorical_crossentropy", metrics=METRICS)

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

    history = model.fit(train_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID, epochs=25, class_weight=weights,
                        callbacks=[lrd, mcp, es])

    return model, history


def plot_metrics(history):
    sns.set()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, acc, color='green', label='Training Accuracy')
    plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, color='pink', label='Training Loss')
    plt.plot(epochs, val_loss, color='red', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def evaluate_model(model_path, test_generator):
    model_eval = tf.keras.models.load_model(model_path)
    hist_eval = model_eval.evaluate(test_generator, verbose=1)
    print(hist_eval)
    print("Accuracy: %f\nLoss: %f" % ((hist_eval[1] * 100), hist_eval[0]))

def Performance_Measures(test_generator):
    pred = model.predict(test_generator)
    con_mat = tf.math.confusion_matrix(labels=test_generator.classes, predictions=np.argmax(pred, axis=1)).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm,
                     index = test_generator.class_indices.keys(),
                     columns = test_generator.class_indices.keys())
    print(con_mat_df)
    import sklearn.metrics
    report = sklearn.metrics.classification_report(test_generator.classes, np.argmax(pred, axis=1), target_names=test_generator.class_indices.keys())
    print(report)
    specificities = []
    for i in range(len(con_mat)):
        tn = np.sum(np.delete(np.delete(con_mat, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(con_mat, i, axis=0)[:, i])
        specificity = tn / (tn + fp)
        specificities.append(specificity)
    for i, specificity in enumerate(specificities):
        print(str(i)+" : "+str(specificity))

Train_img, Train_csv, Test_img, Test_csv, traindf, testdf = load_data()
weights = calculate_class_weights(traindf)
files = os.listdir(Train_img)
preprocessing(files)
train_generator, valid_generator, test_generator = create_data_generators(Train_img, traindf, Test_img, testdf)
model, history = build_model(train_generator,valid_generator,traindf,weights)
plot_metrics(history)
evaluate_model('D:\Materials\Graduation Project\Models\DenseNet121.h5', test_generator)
Performance_Measures(test_generator)