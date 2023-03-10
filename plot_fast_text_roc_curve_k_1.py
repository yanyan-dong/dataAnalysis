import fasttext
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


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
        model_top_4_preds = []
        model_top_1_preds = []
        model_pred_probs = []
        # append the first label to the input ?
        for first_label, text in zip(first_expert_label, example_texts):
            pred = model.predict(f"{first_label} {text}", k=len(label_index_name))
            pred_labels = list(pred[0])
            model_top_4_preds.append(pred_labels[:4])
            model_top_1_preds.append(pred_labels[0])
            # reorder the probs
            top_label_indexes = [label_name_index[i] for i in pred_labels]
            pred_label_probs = list(dict(sorted(dict(zip(top_label_indexes, pred[1].tolist())).items())).values())
            model_pred_probs.append(pred_label_probs)
        model_pred_probs = np.array(model_pred_probs)

        # Evaluate the F1 score
        print(f"{expert} F1 replicated (first expert label) = {f1_score(y_true=first_expert_label, y_pred=model_top_1_preds, average='micro')}")

        plt.rcParams["figure.figsize"] = [8.4, 6.8]
        # binarize the labels and plot each of the class
        first_expert_label_one_hot = label_binarize([label_name_index[i] for i in first_expert_label],
                                                    classes=[i for i in range(len(label_index_name))])
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
        for class_i, (class_name, color) in enumerate(zip(label_index_name, colors)):
            fpr, tpr, _ = roc_curve(first_expert_label_one_hot[:, class_i],
                                    model_pred_probs[:, class_i])
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
        plt.title(f'ROC for {expert} (k = 1)')
        plt.legend(loc="lower right")
        # plt.show()
        # plt.savefig(f'/home/yanyandong/Code/pythonProject/roc_images/roc_{expert}_k_1.png')
        # plt.clf()

        # plot the micro f1 for k = 1
        mirco_fpr, mirco_tpr, _ = roc_curve(first_expert_label_one_hot.ravel(),
                                            model_pred_probs.ravel())
        micro_roc_auc = auc(mirco_fpr, mirco_tpr)
        plt.title(f'Micro Avg. ROC (use first expert label) for {expert} ')
        plt.plot(mirco_fpr, mirco_tpr, 'b', label='AUC = %0.2f' % micro_roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        # plt.show()
        # plt.savefig(f'/home/yanyandong/Code/pythonProject/roc_images/micro_roc_{expert}_k_1.png')
        # plt.clf()