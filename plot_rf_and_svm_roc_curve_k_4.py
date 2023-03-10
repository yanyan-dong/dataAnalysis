import pickle

import nltk
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

from functions_from_previous_works import utils_preprocess_text, oneHotEncoding


def check_missing_labels(labels_list):
    total_num_missing_labels = 0
    for label in labels_list:
        if label == 'NaN':
            total_num_missing_labels += 1
    return total_num_missing_labels


model_path = "/home/yanyandong/Code/pythonProject/trained_models/SVM/svmModelWithBestFit.sav"
vectorizer_path = "/home/yanyandong/Code/pythonProject/trained_models/SVM/vectorizerTFIDFSVM.pickle"
# model_path = "/home/yanyandong/Code/pythonProject/trained_models/Random_Forest/rfModel.sav"
# vectorizer_path = "/home/yanyandong/Code/pythonProject/trained_models/Random_Forest/vectorizerTFIDFRF.pickle"
target_domain_data_path = "/home/yanyandong/Code/pythonProject/data/interaction.labeled.csv"

if __name__ == "__main__":
    # read the dataset
    target_domain_data = pd.read_csv(target_domain_data_path)
    target_domain_data.fillna("NaN", inplace=True)
    example_texts = target_domain_data['DESC'].tolist()
    reed_labels = target_domain_data['Label (R Reed)'].tolist()
    brown_labels = target_domain_data['Label (L Brown)'].tolist()
    glenna_labels = target_domain_data['Label (Glenna)'].tolist()

    all_expert_labels = {'reed': reed_labels,
                         'brown': brown_labels,
                         'glenna': glenna_labels}

    assert len(reed_labels) == len(brown_labels) == len(glenna_labels)
    print("Check missing data in three expert's labels")
    print(f"Number of missing label in reed labels = {check_missing_labels(reed_labels)}")
    print(f"Number of missing label in brown labels = {check_missing_labels(brown_labels)}")
    print(f"Number of missing label in glenna labels = {check_missing_labels(glenna_labels)}")
    # preprocess the text following the previous methods
    lst_stopwords = nltk.corpus.stopwords.words("english")
    processed_text_examples = []
    for text in example_texts:
        processed_text_examples.append(utils_preprocess_text(text,
                                                             flg_stemm=False,
                                                             flg_lemm=True,
                                                             lst_stopwords=lst_stopwords))

    # load the vectorizers and models
    with open(vectorizer_path, 'rb') as f:
        svm_vectorizer = pickle.load(f)
    with open(model_path, 'rb') as f:
        svm_model = pickle.load(f)
    label_index_to_label_name = svm_model.classes_.tolist()
    label_name_to_label_index = {name: i for i, name in enumerate(label_index_to_label_name)}
    tfidf = svm_vectorizer.transform(processed_text_examples)
    model_preds = svm_model.predict(tfidf)
    preds_probs = svm_model.predict_proba(tfidf)
    preds_top_4 = np.argsort(-preds_probs, axis=1)[:, :4]

    # evaluate the svm model as described in previous method and match the number of in the paper
    for expert in ['reed', 'brown', 'glenna']:
        # Get the expert labels
        expert_labels = all_expert_labels[expert]

        # convert the labels to one hot
        expert_labels_one_hot = []
        for label in expert_labels:
            one_hot_label = [0] * len(label_index_to_label_name)
            for sub_label in filter(len, label.split('.')):
                index_ = label_name_to_label_index[sub_label.upper()]
                one_hot_label[index_] = 1
            expert_labels_one_hot.append(one_hot_label)
        expert_labels_one_hot = np.array(expert_labels_one_hot)

        # convert the pred top 4 to one hot
        preds_top_4_one_hot = []
        for label in preds_top_4:
            one_hot_label = [0] * len(label_index_to_label_name)
            for sub_label in label:
                one_hot_label[sub_label] = 1
            preds_top_4_one_hot.append(one_hot_label)

        # Evaluate the F1 score
        print(f"{expert} F1 replicated (k = 4) = {f1_score(y_true=expert_labels_one_hot, y_pred = preds_top_4_one_hot, average = 'micro')}")
        # classification report
        print('Classification report (k = 4):')
        print(classification_report(y_true=expert_labels_one_hot, y_pred=preds_top_4_one_hot,
                                    target_names=label_index_to_label_name, digits=3))

        plt.rcParams["figure.figsize"] = [8.4, 6.8]
        # Plot the ROC curves for each class
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
        for class_i, (class_name, color) in enumerate(zip(label_index_to_label_name, colors)):
            fpr, tpr, _ = roc_curve(expert_labels_one_hot[:, class_i],
                                    preds_probs[:, class_i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr,
                     tpr,
                     color=color,
                     lw=2,
                     label=f'ROC curve of {class_name} (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC (All Expert Labels) for {expert}')
        plt.legend(loc="lower right")
        # plt.show()
        # plt.savefig(f'/home/yanyandong/Code/pythonProject/roc_images/roc_{expert}_k_4.png')
        # plt.clf()

        # plot the micro f1 for k = 1
        mirco_fpr, mirco_tpr, _ = roc_curve(expert_labels_one_hot.ravel(),
                                            preds_probs.ravel())
        micro_roc_auc = auc(mirco_fpr, mirco_tpr)
        plt.title(f'Micro Avg. ROC (All Expert Labels) for {expert} ')
        plt.plot(mirco_fpr, mirco_tpr, 'b', label='AUC = %0.2f' % micro_roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        # plt.show()
        # plt.savefig(f'/home/yanyandong/Code/pythonProject/roc_images/micro_roc_{expert}_k_4.png')
        # plt.clf()