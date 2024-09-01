from utils import *
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q])
        p_a = sigmoid(x)
        if data["is_correct"][i] == 0:
            log_lklihood += np.log(1 - p_a)
        else:
            log_lklihood += np.log(p_a)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    sparse_matrix = data.toarray()
    temp = sigmoid(theta - beta.T)
    grad_theta = np.nanmean(sparse_matrix - temp, axis=1)
    grad_theta = grad_theta.reshape(542, 1)

    grad_beta = -np.nanmean(sparse_matrix - temp, axis=0)
    grad_beta = grad_beta.reshape(1774, 1)

    beta += lr * np.array(grad_beta)
    theta += lr * np.array(grad_theta)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst, neg_lld_list)
    """
    theta = np.zeros((np.shape(data)[0], 1))
    beta = np.zeros((np.shape(data)[1], 1))

    val_acc_lst = []
    neg_lld_list = []
    for i in range(iterations):
        neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        neg_lld_list.append(neg_lld)
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, neg_lld_list


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred)))/len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse(root_dir="../data")
    val_data = load_valid_csv(root_dir="../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    rate_list = []
    iteration_list = []
    accuracy_list = []
    for rate in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]:
        for iteration in [350, 450, 550]:
            a, b, accuracy, neg_lld_list = irt(sparse_matrix, val_data, rate, iteration)
            accuracy_array = np.array(accuracy)
            final = np.max(accuracy_array)
            print(
                "learning rate: {} \t iteration: {} \t accuracy: {}".format(rate, iteration, final))
            rate_list.append(rate)
            iteration_list.append(iteration)
            accuracy_list.append(final)

    max_accuracy = np.max(np.array(accuracy_list))
    i_max = np.argmax(np.array(accuracy_list))
    b_rate = rate_list[i_max]
    b_iteration = iteration_list[i_max]
    print(
        "max validation accuracy: {} \t iteration: {} \t rate: {}".format(max_accuracy, b_iteration,
                                                                          b_rate))

    train_NLLK = irt(sparse_matrix, train_data, b_rate, b_iteration)[3]
    valid_NLLK = irt(sparse_matrix, val_data, b_rate, b_iteration)[3]

    plt.figure()
    plt.title("Curve Showing Training log likelihood at each Iteration")
    plt.plot([x1 for x1 in range(b_iteration)], [-y1 for y1 in train_NLLK],
             label="Training log likelihood")
    plt.xlabel("Iterations")
    plt.ylabel("Log_likelihood")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Curve Showing Validation log likelihood at each Iteration")
    plt.plot([x2 for x2 in range(b_iteration)], [-y2 for y2 in valid_NLLK], color='red',
             label="Validation log likelihood")
    plt.xlabel("Iterations")
    plt.ylabel("Log_likelihood")
    plt.legend()
    plt.show()

    test_accuracy = np.array(irt(sparse_matrix, test_data, b_rate, b_iteration)[2])[-1]
    print("test accuracy: {}".format(test_accuracy))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    theta_array = np.array(irt(sparse_matrix, test_data, b_rate, b_iteration)[0])
    beta_array = np.array(irt(sparse_matrix, test_data, b_rate, b_iteration)[1])
    p1030 = sigmoid(theta_array - beta_array[1030 - 1])
    p1525 = sigmoid(theta_array - beta_array[1525 - 1])
    p1574 = sigmoid(theta_array - beta_array[1574 - 1])

    plt.figure()
    plt.title("Correct Probability v.s. ability")
    plt.plot(theta_array, p1030, 'o', label="question 1030")
    plt.plot(theta_array, p1525, 'o', label="question 1525")
    plt.plot(theta_array, p1574, 'o', label="question 1574")
    plt.legend()
    plt.xlabel("ability")
    plt.ylabel("Correct Probability")
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
