import matplotlib.pyplot as plt


def plot_history(history):
    fig, (ax1, ax0) = plt.subplots(1, 2, figsize=(12, 4))

    # plot
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    ax0.plot(epochs, acc, 'r', label='Training accuracy')
    ax0.plot(epochs, val_acc, 'b', label='Validation accuracy')
    ax0.set_title('Training and validation accuracy')
    ax0.legend(["training acc", "validation acc"])
    ax0.set(xlabel="iteration", ylabel="accuracy")

    ax1.plot(epochs, loss, 'r', label='Training Loss')
    ax1.plot(epochs, val_loss, 'b', label='Validation Loss')
    ax1.set_title('Training and validation loss')
    ax1.legend(["training loss", "validation loss"])
    ax1.set(xlabel="iteration", ylabel="loss")

    return fig
