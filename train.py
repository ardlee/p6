import numpy as np
from preprocess import get_datasets
from models.basic_model import BasicModel
from models.model import Model
from config import image_size
import matplotlib.pyplot as plt
import time
from models.transfered_model import TransferedModel

input_shape = (image_size[0], image_size[1], 3)
categories_count = 3

models = {
    'basic_model': BasicModel,
    'transfered_model': TransferedModel
}

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize = (24, 6))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'b', label = 'Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'b', label = 'Training Loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()

def optimize_hyperparameters(model_class, param_grid, train_dataset, validation_dataset, test_dataset):
    best_accuracy = 0
    best_hyperparameters = None

    # Iterate over hyperparameter grid
    for params in param_grid:
        model = model_class(input_shape, categories_count, **params)
        history = model.train_model(train_dataset, validation_dataset, epochs=30)  # Assuming 30 epochs

        # Evaluate model on validation set
        accuracy = model.evaluate(validation_dataset)

        # Check if current hyperparameters lead to better performance
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hyperparameters = params

    print("Best Hyperparameters:", best_hyperparameters)

    # Train the final model using the best hyperparameters
    best_model = model_class(input_shape, categories_count, **best_hyperparameters)
    best_model.train_model(train_dataset, validation_dataset, epochs=30)  # Assuming 30 epochs

    # Evaluate the final model
    test_accuracy = best_model.evaluate(test_dataset)
    print("Test Accuracy:", test_accuracy)

if __name__ == "__main__":
    # if you want to load your model later, you can use:
    # model = Model.load_model("name_of_your_model.keras")
    # to load your history and plot it again, you can use:
    # history = np.load('results/name_of_your_model.npy',allow_pickle='TRUE').item()
    # plot_history(history)
    # 
    # Your code should change the number of epochs
    epochs = 30
    print('* Data preprocessing')
    train_dataset, validation_dataset, test_dataset = get_datasets()
    name = 'basic_model'
    model_class = models[name]
    print('* Training {} for {} epochs'.format(name, epochs))
    model = model_class(input_shape, categories_count)
    model.print_summary()
    history = model.train_model(train_dataset, validation_dataset, epochs)
    print('* Evaluating {}'.format(name))
    model.evaluate(test_dataset)
    print('* Confusion Matrix for {}'.format(name))
    print(model.get_confusion_matrix(test_dataset))
    model_name = '{}_{}_epochs_timestamp_{}'.format(name, epochs, int(time.time()))
    filename = 'results/{}.keras'.format(model_name)
    model.save_model(filename)
    np.save('results/{}.npy'.format(model_name), history)
    print('* Model saved as {}'.format(filename))
    plot_history(history)


    print('* Data preprocessing')
    train_dataset, validation_dataset, test_dataset = get_datasets()

    # Define hyperparameter grid for each model
    param_grids = {
        'basic_model': [
            {'num_conv_layers': 2, 'num_fc_layers': 1, 'dropout_rate': 0.2, 'learning_rate': 0.001},
            {'num_conv_layers': 3, 'num_fc_layers': 2, 'dropout_rate': 0.3, 'learning_rate': 0.01},
            # Add more hyperparameter combinations as needed
        ],
        'transfered_model': [
            {'num_conv_layers': 2, 'num_fc_layers': 1, 'dropout_rate': 0.2, 'learning_rate': 0.001},
            {'num_conv_layers': 3, 'num_fc_layers': 2, 'dropout_rate': 0.3, 'learning_rate': 0.01},
            # Add more hyperparameter combinations as needed
        ]
    }

    for name, model_class in models.items():
        print('* Optimizing hyperparameters for {}'.format(name))
        optimize_hyperparameters(model_class, param_grids[name], train_dataset, validation_dataset, test_dataset)

# import numpy as np
# from preprocess import get_datasets
# from models.basic_model import BasicModel
# from models.transfered_model import TransferedModel
# from models.model import Model
# from config import image_size
# import matplotlib.pyplot as plt
# import time

# input_shape = (image_size[0], image_size[1], 3)
# categories_count = 3

# models = {
#     'transfered_model': TransferedModel,
# }

# def plot_history(history):
#     acc = history.history['accuracy']
#     val_acc = history.history['val_accuracy']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']

#     epochs = range(1, len(acc) + 1)

#     plt.figure(figsize = (24, 6))
#     plt.subplot(1,2,1)
#     plt.plot(epochs, acc, 'b', label = 'Training Accuracy')
#     plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
#     plt.grid(True)
#     plt.legend()
#     plt.xlabel('Epoch')

#     plt.subplot(1,2,2)
#     plt.plot(epochs, loss, 'b', label = 'Training Loss')
#     plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
#     plt.grid(True)
#     plt.legend()
#     plt.xlabel('Epoch')
#     plt.show()

# if __name__ == "__main__":
#     # if you want to load your model later, you can use:
#     # model = Model.load_model("name_of_your_model.keras")
#     # to load your history and plot it again, you can use:
#     # history = np.load('results/name_of_your_model.npy',allow_pickle='TRUE').item()
#     # plot_history(history)
#     # 
#     # Your code should change the number of epochs
#     epochs = 30
#     print('* Data preprocessing')
#     train_dataset, validation_dataset, test_dataset = get_datasets()
#     name = 'transfered_model'
#     model_class = models[name]
#     print('* Training {} for {} epochs'.format(name, epochs))
#     model = model_class(input_shape, categories_count)
#     model.print_summary()
#     history = model.train_model(train_dataset, validation_dataset, epochs)
#     print('* Evaluating {}'.format(name))
#     model.evaluate(test_dataset)
#     print('* Confusion Matrix for {}'.format(name))
#     print(model.get_confusion_matrix(test_dataset))
#     model_name = '{}_{}_epochs_timestamp_{}'.format(name, epochs, int(time.time()))
#     filename = 'results/{}.keras'.format(model_name)
#     model.save_model(filename)
#     np.save('results/{}.npy'.format(model_name), history)
#     print('* Model saved as {}'.format(filename))
#     plot_history(history)
