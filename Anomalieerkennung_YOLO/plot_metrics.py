import matplotlib.pyplot as plt


def plot_loss(yolo_history):

    plt.figure()

    # plot Yolo loss
    plt.plot(yolo_history['                  epoch'],
             yolo_history['             train/loss'], '-b', label='YOLO train loss')
    plt.plot(yolo_history['                  epoch'],
             yolo_history['               val/loss'], '--b', label='YOLO val loss')



    plt.grid()
    plt.title('Loss vs epochs')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig("loss.png")


def plot_accuracy(yolo_history):

    plt.figure()

    # plot YOLO accuracy
    plt.plot(yolo_history['                  epoch'],
             yolo_history['  metrics/accuracy_top1'] * 100, '-b', label="YOLO Accuracy")


    plt.grid()
    plt.title('Validation accuracy vs epochs')
    plt.ylabel('accuracy (%)')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig("accuracy.png")




