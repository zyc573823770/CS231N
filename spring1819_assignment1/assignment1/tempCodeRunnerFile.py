
# k_choices = tqdm(k_choices)
# for choosen_k in k_choices:
#     k_to_accuracies[choosen_k] = list()
#     for fold_ind in range(num_folds):
#         fold_classifier = KNearestNeighbor()
#         fold_classifier.train(np.delete(X_train, fold_ind, axis=0), np.delete(y_train, fold_ind, axis=0))
#         fold_dist = fold_classifier.compute_distances_two_loops(X_train[fold_ind])
#         fold_predict = fold_classifier.predict_labels(fold_dist, choosen_k)
#         k_to_accuracies[choosen_k].append((float)(np.sum(fold_predict==y_train[fold_ind]))/fold_predict.shape[0])
#         print("val=%d  k=%d  acc=%f"%(fold_ind, choosen_k, (float)(np.sum(fold_predict==y_train[fold_ind]))/fold_predict.shape[0]))




# # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# with open("result.txt") as f:
#     # Print out the computed accuracies
#     for k in sorted(k_to_accuracies):
#         for accuracy in k_to_accuracies[k]:
#             f.writelines('k = %d, accuracy = %f\n' % (k, accuracy))
#             print('k = %d, accuracy = %f' % (k, accuracy))