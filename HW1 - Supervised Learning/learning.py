# decision tree
from sklearn import tree

# neural network
from sklearn.neural_network import MLPClassifier

# (adaptive) boosting
from sklearn.ensemble import AdaBoostClassifier

# k nearest neighbors
from sklearn.neighbors import KNeighborsClassifier

# train and test models using (k-fold) cross-validation
from sklearn.model_selection import cross_val_score

# svm
# from sklearn import svm
from sklearn.svm import SVC

import sys
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    def dec_trees(data_name, train_data, test_data, train_label, test_label):
        max_tree_depths = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59]
        mean_cross_val_scores, train_score, test_score, durations = [], [], [], []

        # want to compare accuracy based on dif amount of pruning
        i = 1;
        for max_tree_depth in range(1, 60, 2):           # runs 30 times
            start_time = time.time()

            classifier = tree.DecisionTreeClassifier(max_depth = max_tree_depth)
            cross_val_scores = cross_val_score(classifier, train_data, train_label, cv = 5)
            mean = 0
            for score in cross_val_scores:
                mean = mean + score
            mean_cross_val_scores.append(mean / len(cross_val_scores))

            classifier = classifier.fit(train_data, train_label)
            duration = time.time() - start_time
            durations.append(duration)

            train_set_accuracy = classifier.score(train_data, train_label)
            train_score.append(train_set_accuracy)
            test_set_accuracy = classifier.score(test_data, test_label)
            test_score.append(test_set_accuracy)

            i = i + 1

        # Create graph 1
        print("*&*&*&*!&&*&*&")
        print(train_label)
        skplt.estimators.plot_learning_curve(classifier,
                                             train_data,
                                             train_label,
                                             title='Learning Curve (' + data_name + ' Decision Trees)')
        plt.savefig('graphs/' + data_name + ' Learning_Curve_(Decision_Trees).png', bbox_inches='tight')
        plt.show()

        # Create graph 2
        plt.figure()
        plt.title('Max Tree Depth vs. Accuracy for ' + data_name + ' decision trees')
        graph_lines = [(mean_cross_val_scores, ':', 'Mean cross-validation score'),
                       (train_score, '--', 'Training set score'),
                       (test_score, '-', 'Testing set score')]
        plt.xlabel('Maximum depths of decision trees')
        plt.ylabel('Accuracy of model')

        for data, line_type, line_label in graph_lines:
            plt.plot(max_tree_depths, data, line_type, label=line_label)
        plt.legend(loc='lower left')
        plt.savefig('graphs/' + data_name + ' decision_trees_Max_Tree_Depth_vs_Accuracy.png', bbox_inches='tight')
        plt.show()

        # Create graph 3
        plt.figure()
        plt.title('Max Tree depths vs. Runtime for ' + data_name + ' decision trees')
        plt.plot(max_tree_depths, durations, '.', label='runtime', color='green')
        plt.xlabel('Maximum depths of decision trees')
        plt.ylabel('Runtime')
        plt.savefig('graphs/' + data_name + ' decision_trees_Max_Tree_Depth_vs_Runtime.png', bbox_inches='tight')
        plt.show()




    # An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases
    def adaboost(data_name, train_data, test_data, train_label, test_label):
        num_estimators = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59]
        max_tree_depth = 13          # accuracy seems to peak at max_depth=11
        max_tree_depth_str = '13'
        mean_cross_val_scores, train_score, test_score, durations = [], [], [], []

        # want to compare accuracy based on if dif # of estimators are used
        i = 1;
        for num_estimator in range(1, 60, 2):
            start_time = time.time()

            tree_classifier = tree.DecisionTreeClassifier(max_depth = max_tree_depth)
            classifier = AdaBoostClassifier(base_estimator = tree_classifier,
                                            n_estimators=(num_estimator))
            cross_val_scores = cross_val_score(classifier, train_data, train_label, cv = 5)
            mean = 0
            for score in cross_val_scores:
                mean = mean + score
            mean_cross_val_scores.append(mean / len(cross_val_scores))

            classifier = classifier.fit(train_data, train_label)
            duration = time.time() - start_time
            durations.append(duration)

            train_set_accuracy = classifier.score(train_data, train_label)
            train_score.append(train_set_accuracy)
            test_set_accuracy = classifier.score(test_data, test_label)
            test_score.append(test_set_accuracy)

            i = i + 1

        # Create graph 1
        skplt.estimators.plot_learning_curve(classifier,
                                             train_data,
                                             train_label,
                                             title='Learning Curve (' + data_name + ' Adaptive Boosting)')
        plt.savefig('graphs/' + data_name + ' Learning_Curve_(Adaptive_Boosting)_maxdepth' + max_tree_depth_str + '.png', bbox_inches='tight')
        plt.show()

        # Create graph 2
        plt.figure()
        plt.title('Number of Estimators vs. Accuracy for ' + data_name + ' adaptive boosting')
        graph_lines = [(mean_cross_val_scores, ':', 'Mean cross-validation score'),
                       (train_score, '--', 'Training set score'),
                       (test_score, '-', 'Testing set score')]
        plt.xlabel('Number of Estimators')
        plt.ylabel('Accuracy of model')

        for data, line_type, line_label in graph_lines:
            plt.plot(num_estimators, data, line_type, label=line_label)
        plt.legend(loc='lower left')
        plt.savefig('graphs/' + data_name + ' adaptive_boosting_Number_of_Estimators_vs_Accuracy_maxdepth' + max_tree_depth_str + '.png', bbox_inches='tight')
        plt.show()

        # Create graph 3
        plt.figure()
        plt.title('Number of Estimators vs. Runtime for ' + data_name + ' adaptive boosting')
        plt.plot(num_estimators, durations, '.', label='runtime', color='green')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Runtime')
        plt.savefig('graphs/' + data_name + ' adaptive_boosting_Number_of_Estimators_vs_Runtime_maxdepth' + max_tree_depth_str + '.png', bbox_inches='tight')
        plt.show()



    # Multi-layer Perceptron classifierself.
    # This model optimizes the log-loss function using LBFGS or stochastic gradient descent. (Logarithmic loss (related to cross-entropy) measures the performance of a classification model where the prediction input is a probability value between 0 and 1. The goal of our machine learning models is to minimize this value.)
    def neural_net(data_name, train_data, test_data, train_label, test_label):
        num_layers = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59]
        mean_cross_val_scores, train_score, test_score, durations = [], [], [], []

        # # want to compare accuracy based on if dif # of layers are used, each layer has same # of neurons (10)
        i = 1;
        for num_layer in range(1, 60, 2):
            classifier_layers = [10] * num_layer
            start_time = time.time()

            classifier = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(classifier_layers), solver='sgd')
            cross_val_scores = cross_val_score(classifier, train_data, train_label, cv = 5)
            mean = 0
            for score in cross_val_scores:
                mean = mean + score
            mean_cross_val_scores.append(mean / len(cross_val_scores))

            classifier = classifier.fit(train_data, train_label)
            duration = time.time() - start_time
            durations.append(duration)

            train_set_accuracy = classifier.score(train_data, train_label)
            train_score.append(train_set_accuracy)
            test_set_accuracy = classifier.score(test_data, test_label)
            test_score.append(test_set_accuracy)

            i = i + 1


        classifier_layers = [10] * 7
        classifier = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(classifier_layers), solver='adam')
        # Create graph 1
        skplt.estimators.plot_learning_curve(classifier,
                                             train_data,
                                             train_label,
                                             title='Learning Curve (' + data_name + ' Neural Networks)')
        plt.savefig('graphs/' + data_name + ' Learning_Curve_(Neural_Networks).png', bbox_inches='tight')
        plt.show()

        # Create graph 2
        plt.figure()
        plt.title('Number of Layers vs. Accuracy for ' + data_name + ' neural network')
        graph_lines = [(mean_cross_val_scores, ':', 'Mean cross-validation score'),
                       (train_score, '--', 'Training set score'),
                       (test_score, '-', 'Testing set score')]
        plt.xlabel('Number of Layers')
        plt.ylabel('Accuracy of model')

        for data, line_type, line_label in graph_lines:
            plt.plot(num_layers, data, line_type, label=line_label)
        plt.legend(loc='lower left')
        plt.savefig('graphs/' + data_name + ' neural_network_Number_of_Layers_vs_Accuracy.png', bbox_inches='tight')
        plt.show()

        # Create graph 3
        plt.figure()
        plt.title('Number of Layers vs. Runtime for ' + data_name + ' neural network')
        plt.plot(num_layers, durations, '.', label='runtime', color='green')
        plt.xlabel('Number of Layers')
        plt.ylabel('Runtime')
        plt.savefig('graphs/' + data_name + ' neural_network_Number_of_Layers_vs_Runtime.png', bbox_inches='tight')
        plt.show()



    # A Support Vector Machine (SVM) performs classification by finding the hyperplane that maximizes the margin between the two classes. The extreme points in the data sets that define the hyperplane are the support vectors. SVM is a supervised machine learning algorithm. It can be used for classification or regression problems. It uses a method called the kernel trick to transform your data.
    def support_vec_mach(data_name, train_data, test_data, train_label, test_label):
        kernels = ['rbf', 'linear']
        mean_cross_val_scores, train_score, test_score, durations = [], [], [], []

        # want to compare accuracy based on dif type kernel used
        for i in range(0, len(kernels)):
            start_time = time.time()

            # for g in [0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02]:
            classifier = SVC(kernel = kernels[i], gamma=.5)#, C=1, gamma=1)
            cross_val_scores = cross_val_score(classifier, train_data, train_label, cv = 5)
            mean = 0
            for score in cross_val_scores:
                mean = mean + score
            mean_cross_val_scores.append(mean / len(cross_val_scores))

            classifier = classifier.fit(train_data, train_label)
            duration = time.time() - start_time
            durations.append(duration)

            train_set_accuracy = classifier.score(train_data, train_label)
            train_score.append(train_set_accuracy)
            test_set_accuracy = classifier.score(test_data, test_label)
            test_score.append(test_set_accuracy)

            print(test_score)


            ### Create graph 1
            skplt.estimators.plot_learning_curve(classifier,
                                                 train_data,
                                                 train_label,
                                                 title='Learning Curve (' + data_name + ' SVM '
                                                    + 'with ' + kernels[i] + ' kernel)')
            plt.savefig('graphs/' + data_name + ' Learning_Curve_(SVM_' + kernels[i] + ').png', bbox_inches='tight')
            plt.show()

        print("Kernel Type vs. Accuracy for " + data_name + " SVM")
        print(kernels)
        print(mean_cross_val_scores)
        print(train_score)
        print(test_score)

        # Create graph 3
        plt.figure()
        plt.title('Kernel Type vs. Runtime for ' + data_name + ' SVM')
        plt.plot(kernels, durations, '.', label='runtime', color='green')
        print("*****************Kernel Type vs. Runtime for " + data_name + " SVM*****************")
        print(kernels)
        print(durations)
        plt.xlabel('Kernel Type')
        plt.ylabel('Runtime')
        plt.savefig('graphs/' + data_name + ' neural_network_Number_of_Layers_vs_Runtime.png', bbox_inches='tight')
        plt.show()

        # print("~!~!~!~~~~~~~~~~~~~~")
        # print(len(mean_cross_val_scores))
        # ### Create graph 2
        # plt.figure()
        # plt.title('Kernel Type vs. Accuracy for ' + data_name + ' SVM')
        # plt.xlabel('Kernel Type')
        # plt.ylabel('Accuracy of model')
        #
        # # ax = plt.subplot(111)
        # width = 0.3
        # ind = np.arange(3)
        # print(mean_cross_val_scores)
        # bar1 = plt.bar(ind, mean_cross_val_scores,width=width,color='b',align='center')
        # bar2 = plt.bar(ind+width, train_score,width=width,color='g',align='center')
        # bar3 = plt.bar(ind+width*2, test_score,width=width,color='r',align='center')
        # # plt.set_xticks(ind+width)
        # # plt.set_xticklabels( (kernels[0], kernels[1]) )
        # plt.xticks(np.arange(2), kernels)
        # plt.legend( (bar1[0], bar2[0], bar3[0]),
        #             ('Mean cross-validation score', 'Training set score', 'Testing set score'),
        #             loc='lower left')
        # # def autolabel(rects):
        # #     for rect in rects:
        # #         h = rect.get_height()
        # #         plt.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
        # #                 ha='center', va='bottom')
        # #
        # # autolabel(bar1)
        # # autolabel(bar2)
        # # autolabel(bar3)
        #
        # plt.savefig('graphs/' + data_name + ' SVM_Kernel_Type_vs_Accuracy.png', bbox_inches='tight')
        # plt.show()
        #
        # # graph_lines = [(mean_cross_val_scores, ':', 'Mean cross-validation score'),
        # #                (train_score, '--', 'Training set score'),
        # #                (test_score, '-', 'Testing set score')]
        #
        # # for data, line_type, line_label in graph_lines:
        # #     plt.plot(num_layers, data, line_type, label=line_label)
        # # plt.legend(loc='lower left')
        # # plt.show()



    # Classifier implementing the k-nearest neighbors vote.
    def k_nearest_neighbors(data_name, train_data, test_data, train_label, test_label):
        num_neighbors = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59]
        mean_cross_val_scores, train_score, test_score, durations = [], [], [], []

        # want to compare accuracy based on if dif # of neighbors are used
        i = 1;
        for num_neighbor in range(1, 60, 2):
            start_time = time.time()

            classifier = KNeighborsClassifier(n_neighbors=num_neighbor, weights='uniform')
            # classifier = KNeighborsClassifier(n_neighbors=num_neighbor, weights='distance')
            cross_val_scores = cross_val_score(classifier, train_data, train_label, cv = 5)
            mean = 0
            for score in cross_val_scores:
                mean = mean + score
            mean_cross_val_scores.append(mean / len(cross_val_scores))

            classifier = classifier.fit(train_data, train_label)
            duration = time.time() - start_time
            durations.append(duration)

            train_set_accuracy = classifier.score(train_data, train_label)
            train_score.append(train_set_accuracy)
            test_set_accuracy = classifier.score(test_data, test_label)
            test_score.append(test_set_accuracy)

            i = i + 1

        # Create graph 1
        skplt.estimators.plot_learning_curve(classifier,
                                             train_data,
                                             train_label,
                                             title='Learning Curve (' + data_name + ' K-NN), weights: uniform')
        plt.savefig('graphs/' + data_name + ' Learning_Curve_(K-NN)' + '_weights_uniform.png', bbox_inches='tight')
        plt.show()

        # Create graph 2
        plt.figure()
        plt.title('Number of Neighbors vs. Accuracy for ' + data_name + ' K-NN, weights: uniform')
        graph_lines = [(mean_cross_val_scores, ':', 'Mean cross-validation score'),
                       (train_score, '--', 'Training set score'),
                       (test_score, '-', 'Testing set score')]
        plt.xlabel('Number of Neighbors')
        plt.ylabel('Accuracy of model')

        for data, line_type, line_label in graph_lines:
            plt.plot(num_neighbors, data, line_type, label=line_label)
        plt.legend(loc='center right')
        plt.savefig('graphs/' + data_name + ' K-NN_Number_of_Neighbors_vs_Accuracy' + '_weights_uniform.png', bbox_inches='tight')
        plt.show()

        # Create graph 3
        plt.figure()
        plt.title('Number of Neighbors vs. Runtime for ' + data_name + ' K-NN, weights: uniform')
        plt.plot(num_neighbors, durations, '.', label='runtime', color='green')
        plt.xlabel('Number of Neighbors')
        plt.ylabel('Runtime')
        plt.savefig('graphs/' + data_name + ' K-NN_Number_of_Estimators_vs_Runtime' + '_weights_uniform.png', bbox_inches='tight')
        plt.show()



    data_name = sys.argv[1]
    train_data, test_data, train_label, test_label = [], [], [], []

    if data_name == 'spam':
        raw_data = pd.read_csv('spambase.data', sep = ',', header=None)      # 4601 instances

        data = raw_data.drop([57], axis=1)
        labels = raw_data[[57]]
        train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.2)      # 3680 + 921 instances

        # print("~~~~~~~~~~~~~@)))))))))))))))tr")
        # print(train_data)
        # print(train_label)
        # print("~~~~~~~~~~~~~@)))))))))))))))12")
        # print(test_data)
        # print(test_label)

    elif data_name == 'satellite':
        raw_data = pd.read_csv('satellite.data', sep = ' ', header=None)      # 6435 instances
        data = raw_data.drop([36], axis=1)
        labels = raw_data[[36]]
        print()
        train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.2)      # 5148 + 1287 instances

        # print("~~~~~~~~~~~~~@)))))))))))))))11")
        # print(train_data)
        # print(train_label)
        # print("~~~~~~~~~~~~~@)))))))))))))))12")
        # print(test_data)
        # print(test_label)

    # dec_trees(data_name, train_data, test_data, train_label, test_label)
    # adaboost(data_name, train_data, test_data, train_label, test_label)
    neural_net(data_name, train_data, test_data, train_label, test_label)
    # support_vec_mach(data_name, train_data, test_data, train_label, test_label)
    # k_nearest_neighbors(data_name, train_data, test_data, train_label, test_label)
