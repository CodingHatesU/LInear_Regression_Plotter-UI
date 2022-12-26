import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#Calling Libraries

def build_model(my_learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
    model.compile(optimizer = tf.keras.optimizers.RMSprop(my_learning_rate),
    loss="mean_squared_error",
    metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def train_model(model, feature, label, epochs, batch_size):
    history = model.fit(x=feature,y = label, batch_size = batch_size, epochs = epochs)
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]
    return trained_weight, trained_bias, epochs, rmse

#Defined Build Model

def plot_the_model(trained_weight, trained_bias, feature, label):
    plt.xlabel("feature")
    plt.ylabel("label")
    plt.scatter(feature,label)
    x0 = 0
    y0 = trained_bias
    x1 = feature[-1]
    y1 = trained_bias + (trained_weight * x1)
    plt.plot([x0,x1],[y0,y1],c='r')
    plt.show()

def plot_the_loss_curve(epochs, rmse):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")
    
    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min()*0.97, rmse.max()])
    plt.show()
    
#Defined Plot Model and Plot Loss Curve functions
my_feature = ([])
my_label = ([])
my_batch_size = int(input("Enter the batch size: "))
for i in range (0, my_batch_size):
    f1 = int(input("Enter the feature: "))
    l1 = int(input("Enter the label: "))
    my_feature.append(f1)
    my_label.append(l1)

#Defined feature and label

learning_rate = float(input("Enter the learning rate: "))
epochs = int(input("Enter the number of epochs: "))

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model,my_feature,my_label, epochs, my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)