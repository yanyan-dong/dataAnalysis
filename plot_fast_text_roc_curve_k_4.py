import fasttext
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


model_path = "/home/yanyandong/Code/pythonProject/trained_models/Fast_Text/fasttext_model_small.bin"
target_domain_data_path = "/home/yanyandong/Code/pythonProject/data/interaction.labeled.csv"

if __name__ == "__main__":
    # read the dataset
    target_domain_data = pd.read_csv(target_domain_data_path)
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

    model = fasttext.load_model(model_path)
    label_index_name = list(model.get_labels())
    label_name_index = {name: i for i, name in enumerate(label_index_name)}

    # evaluate the svm model as described in previous method and match the number of in the paper
    for expert in ['reed', 'brown', 'glenna']:
        expert_labels = all_expert_labels[expert]
        expert_labels_one_hot = []
        first_expert_label = []
        for label in expert_labels:
            first_expert_label.append("__label__" + label.split('.')[0].lower())
            one_hot_label = [0] * len(label_index_name)
            for sub_label in filter(len, label.split('.')):
                index_ = label_name_index[f"__label__{sub_label.lower()}"]
                one_hot_label[index_] = 1
            expert_labels_one_hot.append(one_hot_label)
        expert_labels_one_hot = np.array(expert_labels_one_hot)

        # Run the model to get the predictions
        all_model_pred_top4 = []
        all_model_preds_probs = []
        # append the first label to the input ?
        for first_label, text in zip(first_expert_label, example_texts):
            pred = model.predict(f"{first_label} {text}", k=len(label_index_name))
            pred_labels = list(pred[0])
            all_model_pred_top4.append(pred_labels[:4])
            # reorder the probs
            top_label_indexes = [label_name_index[i] for i in pred_labels]
            pred_label_probs = list(dict(sorted(dict(zip(top_label_indexes, pred[1].tolist())).items())).values())
            all_model_preds_probs.append(pred_label_probs)

        all_model_preds_probs = np.array(all_model_preds_probs)

        # convert the predictions to index
        all_pred_top4_one_hot = []
        for pred_top4 in all_model_pred_top4:
            pred_top4_indexes = [label_name_index[i] for i in pred_top4]
            pred_top4_one_hot = [0] * len(label_index_name)
            for i in pred_top4_indexes:
                pred_top4_one_hot[i] = 1
            all_pred_top4_one_hot.append(pred_top4_one_hot)

        # evaluate f1
        print(f"{expert} F1 replicated (k = 4) = {f1_score(y_true=expert_labels_one_hot, y_pred=all_pred_top4_one_hot, average='micro')}")
        # classification report
        print('Classification report (k = 4):')
        print(classification_report(y_true=expert_labels_one_hot, y_pred=all_pred_top4_one_hot,
                                    target_names=label_index_name, digits=3))

        plt.rcParams["figure.figsize"] = [8.4, 6.8]
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
        for class_i, (class_name, color) in enumerate(zip(label_index_name, colors)):
            fpr, tpr, _ = roc_curve(expert_labels_one_hot[:, class_i],
                                    all_model_preds_probs[:, class_i])
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
        plt.title(f'ROC for {expert} (k = 4)')
        plt.legend(loc="lower right")
        # plt.show()
        # plt.savefig(f'/home/yanyandong/Code/pythonProject/roc_images/roc_{expert}_k_4.png')
        # plt.clf()

        # plot the micro f1 for k = 4
        mirco_fpr, mirco_tpr, _ = roc_curve(expert_labels_one_hot.ravel(),
                                            all_model_preds_probs.ravel())
        micro_roc_auc = auc(mirco_fpr, mirco_tpr)
        plt.title(f'Micro Avg. ROC (all expert label) for {expert} ')
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