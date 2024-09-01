# with mini-batch8, with regularizer
from utils import *
from scipy.linalg import sqrtm

import numpy as np

# import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')


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


def als_with_loss1(train_data, val_data, k, lr, num_iteration, lambda_reg,
                   batch_size=8):
    """ Revised version of als for calculating squared loss with the chosen
    hyperparameter.
    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :param lambda_reg: float (regularization strength)
    :param batch_size: int
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
    train_loss = []
    valid_loss = []
    # Perform updates for a specified number of iterations
    for i in range(num_iteration):
        u, z = update_u_z_mini_batch(train_data, lr, u, z, lambda_reg,
                                     batch_size)
        if i % 100 == 0:
            train_loss.append(squared_error_loss(train_data, u, z))
            valid_loss.append(squared_error_loss(val_data, u, z))

    mat = u @ z.T
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, train_loss, valid_loss


def als1(train_data, k, lr, num_iteration, lambda_reg, batch_size=8):
    """ Performs ALS algorithm, here we use the iterative solution - SGD
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :param lambda_reg: float (regularization strength)
    :param batch_size: int
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
        # print(f"Starting iteration {i + 1}/{num_iteration}")
        u, z = update_u_z_mini_batch(train_data, lr, u, z, lambda_reg,
                                     batch_size)
    mat = u @ z.T
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat


def update_u_z_mini_batch(train_data, lr, u, z, lambda_reg, batch_size=8):
    """Return the updated U and Z after applying mini-batch gradient descent for
    matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float (learning rate)
    :param u: 2D matrix (user factors)
    :param z: 2D matrix (item factors)
    :param lambda_reg: float (regularization strength)
    :param batch_size: int (size of each mini-batch)
    :return: (u, z)
    """
    # Shuffle data indices to create random batches
    indices = np.arange(len(train_data["question_id"]))
    np.random.shuffle(indices)

    for start_idx in range(0, len(indices), batch_size):
        end_idx = min(start_idx + batch_size, len(indices))
        batch_indices = indices[start_idx:end_idx]

        grad_u_total = np.zeros(u.shape)
        grad_z_total = np.zeros(z.shape)

        # Accumulate gradients over the batch
        for i in batch_indices:
            c = train_data["is_correct"][i]
            n = train_data["user_id"][i]
            q = train_data["question_id"][i]

            predicted = np.dot(u[n], z[q].T)
            error = c - predicted

            grad_u = -error * z[q] + lambda_reg * u[n]
            grad_z = -error * u[n] + lambda_reg * z[q]

            grad_u_total[n] += grad_u
            grad_z_total[q] += grad_z

        # Average gradients over batch size
        grad_u_average = grad_u_total / len(batch_indices)
        grad_z_average = grad_z_total / len(batch_indices)

        # Update U and Z with averaged gradients
        for i in batch_indices:
            n = train_data["user_id"][i]
            q = train_data["question_id"][i]
            u[n] -= lr * grad_u_average[n]
            z[q] -= lr * grad_z_average[q]

    return u, z


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #

    # Define the rest hyperparameters
    k_values_als = [3, 9]
    lr_options = [0.01, 0.03, 0.05]
    num_iteration_options = [1000]
    lambda_reg_options = [0.01]

    best_k_als = -1
    best_lr = -1
    best_num_iteration = 1000
    best_val_accuracy_als = 0
    best_lambda_reg = None
    reconstructed_matrix_als = None
    print(f"--ALS with mini-batch GD--")

    # Iterate over the hyperparameters
    for lambda_reg in lambda_reg_options:
        for k in k_values_als:
            for lr in lr_options:
                for num_iteration in num_iteration_options:
                    print(f"Tuning hyperparameters for ALS: k={k}, "
                          f"learning rate={lr}, "
                          f"iterations={num_iteration}, "
                          f"lambda_reg={lambda_reg}")
                    reconstructed_matrix_als = als1(train_data, k, lr,
                                                    num_iteration, lambda_reg)
                    curr_val_accuracy_als = \
                        sparse_matrix_evaluate(val_data,
                                               reconstructed_matrix_als)

                    if curr_val_accuracy_als > best_val_accuracy_als:
                        best_k_als = k
                        best_lr = lr
                        best_num_iteration = num_iteration
                        best_val_accuracy_als = curr_val_accuracy_als
                        best_lambda_reg = lambda_reg

    print(f"Best hyperparameters for ALS: k={best_k_als}, "
          f"learning rate={best_lr}, "
          f"iterations={best_num_iteration}, "
          f"lambda_reg = {best_lambda_reg}")
    print(f"Best validation set accuracy for als: {best_val_accuracy_als}")

    # plot
    finalized_mat, train_loss, valid_loss = \
        als_with_loss1(train_data, val_data, best_k_als, best_lr,
                       best_num_iteration, best_lambda_reg)

    test_acc = sparse_matrix_evaluate(test_data, finalized_mat)
    print(f'Corresponding Test Accuracy: {test_acc}')

    iterations = list(range(0, best_num_iteration, 100))

    # Create subplots with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot on the first subplot
    axs[0].plot(iterations, train_loss, label='Training Loss', color='blue')
    axs[0].set_title('Training Loss')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot on the second subplot
    axs[1].plot(iterations, valid_loss, label='Validation Loss', color='orange')
    axs[1].set_title('Validation Loss')
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
