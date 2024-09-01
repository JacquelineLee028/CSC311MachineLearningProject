from utils import *
from scipy.linalg import sqrtm

import numpy as np
import matplotlib.pyplot as plt


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    # Calculate the error
    # print(u.T[n], z[q])
    predicted = np.dot(u[n], z[q].T)

    error = c - predicted

    # Compute the gradients for u and z respectively
    grad_u = -error * z[q]
    grad_z = -error * u[n]

    # Update u and z
    u[n] -= lr * grad_u
    z[q] -= lr * grad_z

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


# Helper function
def als_with_loss(train_data, val_data, k, lr, num_iteration):
    """ Revised version of als for calculating squared loss with the chosen
    hyperparameter.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix, list, list.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    train_loss = []
    valid_loss = []
    # Perform updates for a specified number of iterations
    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        if i % 100 == 0:
            train_loss.append(squared_error_loss(train_data, u, z))
            valid_loss.append(squared_error_loss(val_data, u, z))

    mat = u @ z.T
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, train_loss, valid_loss


def als(train_data, k, lr, num_iteration):
    """ Performs ALS algorithm, here we use the iterative solution - SGD
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Perform updates for a specified number of iterations
    for _ in range(num_iteration):
        # Log the current iteration and potential loss
        # print(f"Starting iteration {i + 1}/{num_iteration}")
        u, z = update_u_z(train_data, lr, u, z)

    # Reconstruct the matrix from u and z
    # mat = np.dot(u, z.T)
    mat = u @ z.T

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #

    # Try different values of k
    k_values = [1, 3, 5, 7, 8, 9, 11, 13]
    svd_results = {}

    # Run SVD with different k values and report the selected k values.
    print(f'--SVD--')
    print(f'Selected k for SVD: {k_values}')

    for k in k_values:
        reconstructed_matrix = svd_reconstruct(train_matrix, k)
        svd_results[k] = reconstructed_matrix

    # Evaluate the model
    validation_results = {}
    test_results = {}

    for k in k_values:
        validation_accuracy = sparse_matrix_evaluate(val_data, svd_results[k])
        test_accuracy = sparse_matrix_evaluate(test_data, svd_results[k])
        validation_results[k] = validation_accuracy
        test_results[k] = test_accuracy

    # Select the best model
    best_k = max(validation_results, key=validation_results.get)
    best_validation_accuracy = validation_results[best_k]
    best_test_accuracy = test_results[best_k]

    # Report validation and test accuracy
    print(f'Best k: {best_k}')
    print(f'Validation Accuracy for best k: {best_validation_accuracy}')
    print(f'Test Accuracy for best k: {best_test_accuracy}')

    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #

    # Define the rest hyperparameters
    k_values_als = [1, 3, 5, 7, 9, 11, 13]
    lr_options = [0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05]
    num_iteration_options = [100, 1000, 10000, 50000, 100000, 200000]

    best_k_als = -1
    best_lr = -1
    best_num_iteration = -1
    best_val_accuracy_als = 0
    reconstructed_matrix_als = None
    print(f'--ALS with SGD--')
    print(f"Selected hyperparameters for ALS: k={k_values_als}, "
          f"learning rate={lr_options}, "
          f"num_iteration={num_iteration_options}")

    # Iterate over the hyperparameters
    for k in k_values_als:
        for lr in lr_options:
            for num_iteration in num_iteration_options:
                reconstructed_matrix_als = als(train_data, k, lr, num_iteration)
                curr_val_accuracy_als = \
                    sparse_matrix_evaluate(val_data, reconstructed_matrix_als)

                if curr_val_accuracy_als > best_val_accuracy_als:
                    best_k_als = k
                    best_lr = lr
                    best_num_iteration = num_iteration
                    best_val_accuracy_als = curr_val_accuracy_als

    print(f"Best hyperparameters for ALS: k={best_k_als}, "
          f"learning rate={best_lr}, "
          f"iterations={best_num_iteration}")
    print(f"Best validation set accuracy for als: {best_val_accuracy_als}")

    # plot
    finalized_mat, train_loss, valid_loss = \
        als_with_loss(train_data, val_data, best_k_als, best_lr,
                      best_num_iteration)

    test_acc = sparse_matrix_evaluate(test_data, finalized_mat)
    print(f'Corresponding Test Accuracy: {test_acc}')

    iterations = list(range(0, best_num_iteration, 100))

    # Create subplots with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot on the first subplot
    axs[0].plot(iterations, train_loss, label='Training Loss', color='blue')
    axs[0].set_title('Training Squared-error Loss')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot on the second subplot
    axs[1].plot(iterations, valid_loss, label='Validation Loss', color='orange')
    axs[1].set_title('Validation Squared-error Loss')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
