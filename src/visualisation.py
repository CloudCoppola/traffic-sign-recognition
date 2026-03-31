import matplotlib.pyplot as plt

def show_sample_images(dataset, class_names, n=25):

    plt.figure(figsize=(10,10))

    for images, labels in dataset.take(1):
        for i in range(n):
            ax = plt.subplot(5, 5, i + i)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    
    plt.show()

def plot_training_history(history):

    # Loss 
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.title("Loss")

    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['accuracy', 'val_accuracy'])
    plt.title("Accuracy")

    plt.show()