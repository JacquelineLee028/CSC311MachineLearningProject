from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt



def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    #     # the best performance and report the test accuracy with the        #
    #     # chosen k*.                                                        #
    #####################################################################
    k_values = [1, 6, 11, 16, 21, 26]
    valid_acc_user = []
    valid_acc_ques = []

    for k in k_values:
        print(f"Evaluating k={k}:")
        print("Using user-based collaborative filtering")
        acc_user = knn_impute_by_user(sparse_matrix, val_data, k)
        valid_acc_user.append(acc_user)

        print("Using item-based collaborative filtering")
        acc_ques = knn_impute_by_item(sparse_matrix, val_data, k)
        valid_acc_ques.append(acc_ques)

    # part(a)
    i_user = np.argmax(valid_acc_user)
    best_k_user, best_valid_acc_user = k_values[i_user], valid_acc_user[i_user]
    test_acc = knn_impute_by_user(sparse_matrix, test_data, best_k_user)
    print(f"User-based: The selected best k value: {best_k_user}")
    print(f"Validation accuracy with k={best_k_user}: {best_valid_acc_user}")
    print(f"Test accuracy with k={best_k_user}: {test_acc}")

    # part(b)
    i_ques = np.argmax(valid_acc_ques)
    best_k_ques, best_valid_acc_ques = k_values[i_ques], valid_acc_user[i_ques]
    test_acc_ques = knn_impute_by_item(sparse_matrix, test_data, best_k_ques)
    print(f"Item-based: The selected best k value: {best_k_ques}")
    print(f"Validation accuracy with k={best_k_ques}: {best_valid_acc_ques}")
    print(f"Test accuracy with k={best_k_ques}: {test_acc_ques}")

    # Uncomment for plot
    plt.plot(k_values, valid_acc_user, label='User-based', color='blue', marker='o')
    plt.plot(k_values, valid_acc_ques, label='Item-based', color='red', marker='x')
    plt.title("Validation Accuracy for a kNN model")
    plt.xlabel("k")
    plt.xticks(k_values)
    plt.ylabel("Accuracy")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
