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

    all_expert_labels = {'expert_1': reed_labels,
                         'expert_3': brown_labels,
                         'expert_2': glenna_labels}

    assert len(reed_labels) == len(brown_labels) == len(glenna_labels)
    print("Check missing data in three expert's labels")
    print(f"Number of missing label in reed labels = {check_missing_labels(reed_labels)}")
    print(f"Number of missing label in brown labels = {check_missing_labels(brown_labels)}")
    print(f"Number of missing label in glenna labels = {check_missing_labels(glenna_labels)}")

    model = fasttext.load_model(model_path)
    label_index_name = list(model.get_labels())
    label_name_index = {name: i for i, name in enumerate(label_index_name)}

    # evaluate the svm model as described in previous method and match the number of in the paper
    plt.title(f'Micro Avg. ROC')
    colors = ['seagreen', 'seagreen', 'red', 'red', 'dodgerblue', 'dodgerblue']
    color_idx = 0
    for expert in ['expert_1', 'expert_2', 'expert_3']:
        expert_labels = all_expert_labels[expert]
        two_expert_labels = []
        for label in expert_labels:
            two_expert_labels.append(label.split('.')[:2])
        three_expert_labels = []
        for label in expert_labels:
            three_expert_labels.append(label.split('.')[:3])
        expert_labels_one_hot = []
        two_expert_labels_one_hot = []
        three_expert_labels_one_hot = []
        first_expert_label = []
        for label in expert_labels:
            first_expert_label.append("__label__" + label.split('.')[0].lower())
            one_hot_label = [0] * len(label_index_name)
            for sub_label in filter(len, label.split('.')):
                index_ = label_name_index[f"__label__{sub_label.lower()}"]
                one_hot_label[index_] = 1
            expert_labels_one_hot.append(one_hot_label)
        expert_labels_one_hot = np.array(expert_labels_one_hot)
        for label in two_expert_labels:
            one_hot_label = [0] * len(label_index_name)
            for sub_label in filter(len, label[:2]):
                index_ = label_name_index[f"__label__{sub_label.lower()}"]
                one_hot_label[index_] = 1
            two_expert_labels_one_hot.append(one_hot_label)
        two_expert_labels_one_hot = np.array(two_expert_labels_one_hot)
        for label in three_expert_labels:
            one_hot_label = [0] * len(label_index_name)
            for sub_label in filter(len, label[:3]):
                index_ = label_name_index[f"__label__{sub_label.lower()}"]
                one_hot_label[index_] = 1
            three_expert_labels_one_hot.append(one_hot_label)
        three_expert_labels_one_hot = np.array(three_expert_labels_one_hot)


        # Run the model to get the predictions
        model_top_4_preds = []
        model_top_1_preds = []
        model_pred_probs = []
        for text in example_texts:
            pred = model.predict(f"{text}", k=len(label_index_name))
            pred_labels = list(pred[0])
            model_top_4_preds.append(pred_labels[:4])
            model_top_1_preds.append(pred_labels[0])
            # reorder the probs
            top_label_indexes = [label_name_index[i] for i in pred_labels]
            pred_label_probs = list(dict(sorted(dict(zip(top_label_indexes, pred[1].tolist())).items())).values())
            model_pred_probs.append(pred_label_probs)
        model_pred_probs = np.array(model_pred_probs)

        # binarize the labels and plot each of the class
        first_expert_label_one_hot = label_binarize([label_name_index[i] for i in first_expert_label],
                                                    classes=[i for i in range(len(label_index_name))])


        # convert the predictions to index
        all_pred_top4_one_hot = []
        for pred_top4 in model_top_4_preds:
            pred_top4_indexes = [label_name_index[i] for i in pred_top4]
            pred_top4_one_hot = [0] * len(label_index_name)
            for i in pred_top4_indexes:
                pred_top4_one_hot[i] = 1
            all_pred_top4_one_hot.append(pred_top4_one_hot)

        print(f"{expert} F1 replicated (first expert label) = {f1_score(y_true=first_expert_label, y_pred=model_top_1_preds, average='micro')}")
        print(f"{expert} F1 replicated (k = 4) = {f1_score(y_true=expert_labels_one_hot, y_pred=all_pred_top4_one_hot, average='micro')}")

        # plot the micro f1 for k = 1
        k1_mirco_fpr, k1_mirco_tpr, _ = roc_curve(first_expert_label_one_hot.ravel(),
                                                  model_pred_probs.ravel())
        k1_roc_auc = auc(k1_mirco_fpr, k1_mirco_tpr)
        plt.plot(k1_mirco_fpr,
                 k1_mirco_tpr,
                 color=colors[color_idx],
                 lw=2,
                 label=f'{expert} k = 1 (area = {k1_roc_auc:0.2f})',
                 linestyle='dashed')
        color_idx += 1

        # k4_mirco_fpr, k4_mirco_tpr, _ = roc_curve(expert_labels_one_hot.ravel(),
        #                                           model_pred_probs.ravel())
        # k4_roc_auc = auc(k4_mirco_fpr, k4_mirco_tpr)
        # plt.plot(k4_mirco_fpr,
        #          k4_mirco_tpr,
        #          color=colors[color_idx],
        #          lw=2,
        #          label=f'{expert} k =4 (area = {k4_roc_auc:0.2f})')

        # k2_mirco_fpr, k2_mirco_tpr, _ = roc_curve(two_expert_labels_one_hot.ravel(),
        #                                           model_pred_probs.ravel())
        # k2_roc_auc = auc(k2_mirco_fpr, k2_mirco_tpr)
        # plt.plot(k2_mirco_fpr,
        #          k2_mirco_tpr,
        #          color=colors[color_idx],
        #          lw=2,
        #          label=f'{expert} k = 2 (area = {k2_roc_auc:0.2f})')

        k3_mirco_fpr, k3_mirco_tpr, _ = roc_curve(three_expert_labels_one_hot.ravel(),
                                                  model_pred_probs.ravel())
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
    plt.title(f'FastText Avg. ROC Curve')
    plt.legend(loc="lower right")
    # plt.show()
    # plt.savefig(f'/home/yanyandong/Code/pythonProject/roc_updated/FastText_k3.png', dpi=300)
    # plt.clf()