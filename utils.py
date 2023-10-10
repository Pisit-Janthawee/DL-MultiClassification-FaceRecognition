import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def show_image(images, target, target_names, title, num_display=16, num_cols=4, cmap='gray', random_mode=False):
    '''
    :Parameters
        images (ndarray (n,)): Input data as a numpy array.
        target (ndarray (n,)): Target data as a numpy array.
        title (String): Title of the plot.
        num_display (int): Number of images to display. Default is 16.
        num_cols (int): Number of columns in the plot. Default is 4.
        cmap (str): Color map for displaying images. Default is 'gray'.
        random_mode (bool): If True, display images randomly. If False, display the first num_display images. Default is False.
    '''
    # Determine the number of rows based on the num_cols parameter
    n_cols = min(num_cols, num_display)
    n_rows = int(np.ceil(num_display / n_cols))

    n_images = min(num_display, len(images))
    if random_mode:
        random_indices = np.random.choice(
            len(images), num_display, replace=False)
    else:
        random_indices = np.arange(num_display)

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(20, 4*n_rows))
    for i, ax in enumerate(axes.flatten()):
        if i >= n_images:  # Check if the index exceeds the available number of images
            break
        # Incase (Did PCA)
        index = random_indices[i]
        if len(images.shape) == 2:
            image = images[index].reshape((128, 128)).astype(int)
        else:
            image = images[index]

        ax.imshow(image, cmap=cmap)
        ax.set_title(
            f"Target: {target[index]} ({target_names[target[index]]})" if target_names else f"Target: {target[index]}")

    plt.suptitle(f"{title} (Displaying {num_display} Images)",
                 fontsize=16, fontweight='bold')

    fig.set_facecolor('white')
    plt.tight_layout()
    return plt.show()


def plot_channel_distribution(df, title):
    # Reshape the input data to make it suitable for plotting
    num_images, height, width, channels = df.shape
    # Reshape to (num_images, pixels, channels)
    df_reshaped = df.reshape(num_images, -1, channels)

    # Create a subplot with 1 row and 3 columns (one for each channel)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    channel_names = ['Red', 'Green', 'Blue']

    # Plot the distribution of each channel
    for i in range(channels):
        sns.distplot(df_reshaped[:, :, i].flatten(),
                     bins=20, color=channel_names[i], ax=axes[i])
        axes[i].set_title(f"{channel_names[i]} Channel Distribution")

    # Set the overall title for the subplots
    fig.suptitle(title, fontsize=12)

    plt.show()


def plot_class_distribution(X, y, title, classes, train_percent=0.6, val_percent=0.2, test_percent=0.2):
    '''
    :Parameters:
    - X (numpy array): The input feature matrix of shape (num_examples, num_features).
    - y (numpy array): The target labels of shape (num_examples,).
    - title (str): The title for the entire plot.
    - classes (list): A list of class labels, e.g., ['Normal', 'Tuberculosis'].
    - train_percent (float): Percentage of data for training set.
    - val_percent (float): Percentage of data for validation set.
    - test_percent (float): Percentage of data for test set.

    :Returns:
    None (Displays the plot).
    '''
    assert train_percent + val_percent + \
        test_percent == 1.0, "Sum of train_percent, val_percent, and test_percent should be 1.0"

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_percent+test_percent, random_state=42, stratify=y)

    test_size = test_percent / (val_percent + test_percent)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size, random_state=42, stratify=y_temp)

    # Create a subplot with 3 columns and 1 row
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # List of subset names
    subset_names = ['Train', 'Test', 'Validation']
    n = len(X)

    # Iterate over subsets
    for i, subset in enumerate([(X_train, y_train), (X_test, y_test), (X_val, y_val)]):
        X_subset, y_subset = subset
        n_subset = len(y_subset)
        # Get the class counts
        class_counts = np.bincount(y_subset)

        # Plot histogram for current subset
        axs[i].bar(classes, class_counts, color='#AA99FF')
        subtitle = r'$\bf{' + subset_names[i] + \
            '}$' + f' {int(n_subset/n*100)} %'
        axs[i].set_title(
            subtitle + f'\n Size = {X_subset.shape[0]}', fontsize=18)
        axs[i].set_xlabel('Class')
        axs[i].set_ylabel('Number of examples')

        # Add labels to the bars
        for j, count in enumerate(class_counts):
            axs[i].text(j, count, str(count), ha='center',
                        va='bottom', fontsize=12)

    class_counts = np.bincount(y)
    class_balance_text = ' | '.join(
        [f'{class_label}: {count}' for class_label, count in zip(classes, class_counts)])
    plt.suptitle(f'{title}' + f'\n Training examples (X) = {X.shape[0]}' +
                 f'\n Class balance = {class_balance_text}', fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.show()


def compare_actual_and_predicted(estimator, images, target, class_labels, title, num_display=16, num_cols=4, random_mode=False):
    '''
    Compare Actual Images with Model Predictions for Multiclass Classification.

    Parameters:
        estimator: Model used for predictions.
        images (ndarray): Input data as a numpy array.
        target (ndarray): Target data as a numpy array (true class labels).
        class_labels (list): List of class labels.
        title (str): Title of the plot.
        num_display (int, optional): Number of images to display. Default is 16.
        num_cols (int, optional): Number of columns in the plot. Default is 4.
        random_mode (bool, optional): If True, display images randomly. If False, display the first num_display images. Default is False.

    Returns:
        None

    This function generates a visual comparison between actual images and their corresponding model predictions for multiclass classification. It displays a grid of images with labels to show whether the model's predictions match the actual class labels. The grid is organized in rows and columns based on the specified parameters.

    The function does not return any values; it displays the comparison plot directly.
    '''

    # Determine the number of rows based on the num_cols parameter
    n_cols = min(num_cols, num_display)
    n_rows = int(np.ceil(num_display / n_cols))

    # Get model predictions and class probabilities
    y_pred = estimator.predict(images)
    predicted_labels = np.argmax(y_pred, axis=1)
    predicted_probs = np.max(y_pred, axis=1)

    n_images = min(num_display, len(images))
    if random_mode:
        random_indices = np.random.choice(
            len(images), num_display, replace=False)
    else:
        random_indices = np.arange(num_display)

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(20, 4*n_rows))
    for i, ax in enumerate(axes.flatten()):
        if i >= n_images:  # Check if the index exceeds the available number of images
            break
        index = random_indices[i]
        image = images[index]

        actual_label = class_labels[target[index]]
        model_pred_label = class_labels[predicted_labels[index]]
        model_prob = '{:.3f}'.format(predicted_probs[index])

        ax.imshow(image, cmap=None if actual_label ==
                  model_pred_label else 'OrRd')
        ax.set_title(
            f"Actual: {actual_label}\nPrediction: {model_pred_label}\nProbability: {model_prob}", fontsize=12)

    plt.suptitle(f"{title} (Displaying {num_display} Images)",
                 fontsize=16, fontweight='bold')

    fig.set_facecolor('white')
    plt.tight_layout()
    plt.show()


def compare_actual_and_predicted_prob(estimator, images, target, class_labels, title, num_display=16, random_mode=False):
    '''
    Compare Actual Images with Model Predictions for Multiclass Classification.

    Parameters:
        estimator: Model used for predictions.
        images (ndarray): Input data as a numpy array.
        target (ndarray): Target data as a numpy array (true class labels).
        class_labels (list): List of class labels.
        title (str): Title of the plot.
        num_display (int, optional): Number of images to display. Default is 16.
        random_mode (bool, optional): If True, display images randomly. If False, display the first num_display images. Default is False.

    Returns:
        None

    This function generates a visual comparison between actual images and their corresponding model predictions for multiclass classification. It displays a grid of images in the first column and horizontal bar plots of model predicted probabilities in the second column, showing whether the model's predictions match the actual class labels. The grid is organized in rows based on the specified parameters.

    The function does not return any values; it displays the comparison plot directly.
    '''

    # Determine the number of rows based on the num_cols parameter
    num_cols = 2  # Two columns for actual images and predicted probabilities
    n_cols = min(num_cols, num_display)
    n_rows = int(np.ceil(num_display / n_cols))

    # Get model predictions and class probabilities
    y_pred = estimator.predict(images)
    predicted_labels = np.argmax(y_pred, axis=1)
    predicted_probs = np.max(y_pred, axis=1)

    n_images = min(num_display, len(images))
    if random_mode:
        random_indices = np.random.choice(
            len(images), num_display, replace=False)
    else:
        random_indices = np.arange(num_display)

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(15, 4 * n_rows))

    for i, (ax1, ax2) in enumerate(axes):
        if i >= n_images:  # Check if the index exceeds the available number of images
            break
        index = random_indices[i]
        image = images[index]

        actual_label = class_labels[target[index]]
        model_pred_label = class_labels[predicted_labels[index]]
        model_prob = predicted_probs[index]
        model_prob = '{:.3f}'.format(model_prob)
        ax1.imshow(image, cmap='gray')
        ax1.set_title(
            f"Actual: {actual_label}\nPrediction: {model_pred_label}\nProbability: {model_prob}", fontsize=12)

        # Create a horizontal bar plot for model predicted probabilities
        ax2.barh(class_labels, y_pred[index])
        ax2.set_xlim(0, 1.0)
        ax2.set_xlabel('Probability')
        ax2.set_title('Model Predicted Probabilities', fontsize=12)

        # Add text values on the bars
        for j, prob in enumerate(y_pred[index]):
            ax2.text(prob + 0.01, j, f'{prob:.3f}', va='center', fontsize=10)

    plt.suptitle(f"{title} (Displaying {num_display} Images)",
                 fontsize=16, fontweight='bold')

    fig.set_facecolor('white')
    plt.tight_layout()
    plt.show()
