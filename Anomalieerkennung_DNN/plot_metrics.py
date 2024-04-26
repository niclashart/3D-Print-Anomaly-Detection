import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_loss(cnn_history, mini_cnn_history, save_path):

    plt.figure()

    # plot cnn loss
    plt.plot(cnn_history['loss'], '-r', label='CNN train loss')
    plt.plot(cnn_history['val_loss'], '--r', label='CNN val loss')

    # plot mini cnn loss
    plt.plot(mini_cnn_history['loss'], '-g', label='Mini CNN train loss')
    plt.plot(mini_cnn_history['val_loss'], '--g', label='Mini CNN val loss')

    plt.grid()
    plt.title('Loss vs epochs')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig(save_path)


def plot_accuracy(cnn_history, mini_cnn_history, save_path):

    plt.figure()

    # plot cnn accuracy
    plt.plot(cnn_history['val_accuracy'] * 100, '-r', label="CNN Accuracy")

    # plot mini cnn accuracy
    plt.plot(mini_cnn_history['val_accuracy'] *
             100, "-g", label="Mini CNN Accuracy")

    plt.grid()
    plt.title('Validation accuracy vs epochs')
    plt.ylabel('accuracy (%)')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig(save_path)


def plot_confusion_matrix(test_labels, predicted_labels,save_path):

    confusion_matrix_predicted = confusion_matrix(test_labels, predicted_labels)

    plt.figure()
    plt.title("Confusion Matrix")
    sns.heatmap(confusion_matrix_predicted, annot=True, fmt='d', cmap='Reds')
    plt.savefig(save_path)
