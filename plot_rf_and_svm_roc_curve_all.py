import pickle

import nltk
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

from functions_from_previous_works import utils_preprocess_text, oneHotEncoding


def check_missing_labels(labels_list):
    total_num_missing_labels = 0
    for label in labels_list:
        if label == 'NaN':
            total_num_missing_labels += 1
    return total_num_missing_labels


# model_path = "/home/yanyandong/Code/pythonProject/trained_models/SVM/svmModelWithBestFit.sav"
# vectorizer_path = "/home/yanyandong/Code/pythonProject/trained_models/SVM/vectorizerTFIDFSVM.pickle"
model_path = "/home/yanyandong/Code/pythonProject/trained_models/Random_Forest/rfModel.sav"
vectorizer_path = "/home/yanyandong/Code/pythonProject/trained_models/Random_Forest/vectorizerTFIDFRF.pickle"
target_domain_data_path = "/home/yanyandong/Code/pythonProject/data/interaction.labeled.csv"

if __name__ == "__main__":
    # read the dataset
    target_domain_data = pd.read_csv(target_domain_data_path)
    target_domain_data.fillna("NaN", inplace=True)
    example_texts = target_domain_data['DESC'].tolist()
    reed_labels = target_domain_data['Label (R Reed)'].tolist()
    brown_labels = target_domain_data['Label (L Brown)'].tolist()
    glenna_labels = target_domain_data['Label (Glenna)'].tolist()

    all_expert_labels = {'expert_1': reed_labels,
                         'expert_3': brown_labels,
                         'expert_2': glenna_labels}

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
    preds_top_1 = np.argsort(-preds_probs, axis=1)[:, :1]


    # plot ROC curves
    plt.title(f'Micro Avg. ROC')
    colors = ['seagreen', 'seagreen', 'red', 'red', 'dodgerblue', 'dodgerblue']
    color_idx = 0
    for expert in ['expert_1', 'expert_2', 'expert_3']:
        # Get the expert labels
        expert_labels = all_expert_labels[expert]

        # All expert labels
        expert_labels_one_hot = []
        for label in expert_labels:
            one_hot_label = [0] * len(label_index_to_label_name)
            for sub_label in filter(len, label.split('.')):
                index_ = label_name_to_label_index[sub_label.upper()]
                one_hot_label[index_] = 1
            expert_labels_one_hot.append(one_hot_label)
        expert_labels_one_hot = np.array(expert_labels_one_hot)

        # First Expert labels
        # Only use the first label
        first_expert_label = []
        for label in expert_labels:
            first_expert_label.append(label.split('.')[0])
        binarized_first_expert_label = label_binarize(
            [label_name_to_label_index[i.upper()] for i in first_expert_label],
            classes=[i for i in range(len(label_index_to_label_name))])

        # Only use first two labels
        two_expert_labels = []
        for label in expert_labels:
            two_expert_labels.append(label.split('.')[:2])

        two_expert_labels_one_hot = []
        for label in two_expert_labels:
            one_hot_label = [0] * len(label_index_to_label_name)
            for sub_label in filter(len, label[:2]):
                index_ = label_name_to_label_index[sub_label.upper()]
                one_hot_label[index_] = 1
            two_expert_labels_one_hot.append(one_hot_label)
        two_expert_labels_one_hot = np.array(two_expert_labels_one_hot)

        # Only use first three labels
        three_expert_labels = []
        for label in expert_labels:
            three_expert_labels.append(label.split('.')[:3])

        three_expert_labels_one_hot = []
        for label in three_expert_labels:
            one_hot_label = [0] * len(label_index_to_label_name)
            for sub_label in filter(len, label[:3]):
                index_ = label_name_to_label_index[sub_label.upper()]
                one_hot_label[index_] = 1
            three_expert_labels_one_hot.append(one_hot_label)
        three_expert_labels_one_hot = np.array(three_expert_labels_one_hot)

        k1_mirco_fpr, k1_mirco_tpr, _ = roc_curve(binarized_first_expert_label.ravel(),
                                                  preds_probs.ravel())
        k1_roc_auc = auc(k1_mirco_fpr, k1_mirco_tpr)
        plt.plot(k1_mirco_fpr,
                 k1_mirco_tpr,
                 color=colors[color_idx],
                 lw=2,
                 label=f'{expert} k = 1 (area = {k1_roc_auc:0.2f})',
                 linestyle='dashed')
        color_idx += 1

        # k4_mirco_fpr, k4_mirco_tpr, _ = roc_curve(expert_labels_one_hot.ravel(),
        #                                           preds_probs.ravel())
        # k4_roc_auc = auc(k4_mirco_fpr, k4_mirco_tpr)
        # plt.plot(k4_mirco_fpr,
        #          k4_mirco_tpr,
        #          color=colors[color_idx],
        #          lw=2,
        #          label=f'{expert} k = 4 (area = {k4_roc_auc:0.2f})')

        # k2_mirco_fpr, k2_mirco_tpr, _ = roc_curve(two_expert_labels_one_hot.ravel(),
        #                                           preds_probs.ravel())
        # k2_roc_auc = auc(k2_mirco_fpr, k2_mirco_tpr)
        # plt.plot(k2_mirco_fpr,
        #          k2_mirco_tpr,
        #          color=colors[color_idx],
        #          lw=2,
        #          label=f'{expert} k = 2 (area = {k2_roc_auc:0.2f})')

        k3_mirco_fpr, k3_mirco_tpr, _ = roc_curve(three_expert_labels_one_hot.ravel(),
                                                  preds_probs.ravel())
        k3_roc_auc = auc(k3_mirco_fpr, k3_mirco_tpr)
        plt.plot(k3_mirco_fpr,
                 k3_mirco_tpr,
                 color=colors[color_idx],
                 lw=2,
                 label=f'{expert} k = 3 (area = {k3_roc_auc:0.2f})')

        color_idx += 1
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Random Forest Avg. ROC Curve')
    plt.legend(loc="lower right")
    # plt.show()
    # plt.savefig(f'/home/yanyandong/Code/pythonProject/roc_updated/RF_k3.png', dpi=300)
    # plt.clf()
