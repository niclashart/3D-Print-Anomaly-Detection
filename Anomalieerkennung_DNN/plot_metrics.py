import matplotlib.pyplot as plt


def plot_loss(cnn_history, mini_cnn_history):

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
    plt.savefig("./Anomalieerkennung_DNN/loss.png")


def plot_accuracy(cnn_history, mini_cnn_history):

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
    plt.savefig("./Anomalieerkennung_DNN/accuracy.png")
