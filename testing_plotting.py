from main import train_softmax_batches
from main import train_sigmoid_batches
from main import create_batches
from main import build_network
from activations import Relu
from activations import LeakyRelu
from activations import Tanh
from activations import Sigmoid
from activations import Softmax_CrossEntropyLossConjoined
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('wdbc.data', header=None)
X = df.drop(columns=[0,1])
X.to_numpy()
Y_string = df[1].to_numpy()


# Below code converts labels to integer values were Malignant is 1
i = 0
Y = np.zeros(len(Y_string)).astype(int)
while i <= len(Y_string) - 1:
    if Y_string[i] == 'M':
        Y[i] = 1
    i += 1

int_count = 0
for i in range(len(Y_string)):
    if(Y_string[i] != 'M'):
        int_count += 1

print(int_count / len(Y_string))

# Booleans for testing
plot = True
paramSearch = False

hyper_params = {
    "model_type": "softma1x",
    "ann_architecture": [8, 8, 1],
    "ann_activations": [LeakyRelu(), LeakyRelu(), Sigmoid()],
    "epoch": 200,
    "lr": 0.001,
    "lr_schedule": 1,
    "batch_size": 64,
    "lr_decay": 0.05,
    "lr_momentum": 0.9,
    "lr_drop_freq": 10,
    "lr_drop_rate": 0.5
}

# From hyper params return index of results from model
# Create model
network = build_network(hyper_params["ann_architecture"], hyper_params["ann_activations"])
# Create batches
x_batches, y_batches, step_count = create_batches(hyper_params["batch_size"], X, Y)
print(y_batches[0].shape)

if paramSearch == True:
    paramToSearch = input("Param key: ")
    lower_range = float(input("Lower range: "))
    upper_range = float(input("Upper range: "))
    num_of_values_to_check = int(input("Number of values to check within range: "))
    iterations = int(input("Iterations: "))
    values = np.linspace(start=lower_range, stop=upper_range, num=num_of_values_to_check)
    results_for_value = []
    results_for_value_sd = []
    for value in range(len(values)):
        hyper_params[paramToSearch] = values[value]
        results_for_iterations = []
        for iteration in range(iterations):
            if hyper_params["model_type"] == "softmax":
                results = train_softmax_batches(x_batches=x_batches, y_batches=y_batches, lr=hyper_params["lr"],
                                                lr_schedule=hyper_params["lr_schedule"],
                                                lr_decay=hyper_params["lr_decay"],
                                                network=network, drop_freq=hyper_params["lr_drop_freq"],
                                                drop_rate=hyper_params["lr_drop_rate"], epochs=hyper_params["epoch"],
                                                momentum=hyper_params["lr_momentum"], step_count=step_count)
            else:
                results = train_sigmoid_batches(x_batches=x_batches, y_batches=y_batches, lr=hyper_params["lr"],
                                                lr_schedule=hyper_params["lr_schedule"],
                                                lr_decay=hyper_params["lr_decay"],
                                                network=network, drop_freq=hyper_params["lr_drop_freq"],
                                                drop_rate=hyper_params["lr_drop_rate"], epochs=hyper_params["epoch"],
                                                momentum=hyper_params["lr_momentum"], step_count=step_count)
                results_for_iterations.append(results["acc"][-10:])
        # Get the average accuracy over the iterations
        results_for_value.append(np.mean(results_for_iterations))
        results_for_value_sd.append(np.std(results_for_iterations))
    i = 0
    while i < len(results_for_value):
        print("Av acc for: ", values[i], " was: ", results_for_value[i], " Std: ", results_for_value_sd[i])
        i += 1

if paramSearch == False:
    # Run model and return results for plotting
    if hyper_params["model_type"] == "softmax":
        results = train_softmax_batches(x_batches= x_batches, y_batches= y_batches, lr= hyper_params["lr"],
                                        lr_schedule=hyper_params["lr_schedule"], lr_decay=hyper_params["lr_decay"],
                                        network=network, drop_freq=hyper_params["lr_drop_freq"],
                                        drop_rate=hyper_params["lr_drop_rate"], epochs=hyper_params["epoch"],
                                        momentum=hyper_params["lr_momentum"], step_count=step_count)
    else:
        results = train_sigmoid_batches(x_batches=x_batches, y_batches= y_batches, lr=hyper_params["lr"],
                                        lr_schedule=hyper_params["lr_schedule"], lr_decay=hyper_params["lr_decay"],
                                        network=network, drop_freq=hyper_params["lr_drop_freq"],
                                        drop_rate=hyper_params["lr_drop_rate"], epochs=hyper_params["epoch"],
                                        momentum=hyper_params["lr_momentum"], step_count=step_count)

    # Now we have results can plot
    list_of_epochs = results["epoch"]
    list_of_accs = results["acc"]
    list_of_loss = results["loss"]
    list_of_lrs = results["lr"]


if plot == True:
    # X axis parameter:
    xaxis = list_of_epochs

    # Y axis parameter:
    yaxis = list_of_accs

    plt.plot(xaxis, yaxis)
    plt.suptitle("Lr 0.001 - Batch Size 64")
    title = "Max accuracy achieved: " + str(np.max(list_of_accs))
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.show()
