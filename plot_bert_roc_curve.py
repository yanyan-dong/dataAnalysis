import torch
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
from torch import nn
from sklearn.metrics import roc_curve, auc, f1_score, classification_report, precision_score
from transformers import BertModel, TrainingArguments, Trainer
from transformers import BertTokenizer


def utils_preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    lst_text = text.split()
    text = " ".join(lst_text)
    return text


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 7)  # there are 7 labels
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


class SIRDataset(torch.utils.data.Dataset):
    def __init__(self,
                 examples,
                 pretrained_tokenizer,
                 max_length,
                 labels=None):
        self.examples = examples
        self.labels = labels
        self.pretrained_tokenizer = pretrained_tokenizer
        self.max_length = max_length

    def __getitem__(self, item):
        tokenized_features = self.pretrained_tokenizer(self.examples[item],
                                                       max_length=self.max_length,
                                                       padding='max_length',
                                                       truncation=True,
                                                       return_tensors='pt')
        input_ids = tokenized_features.input_ids.squeeze()
        attention_mask = tokenized_features.attention_mask.squeeze()

        outputs = dict(input_ids=input_ids,
                       attention_mask=attention_mask)

        if self.labels is not None:
            outputs['labels'] = torch.tensor(self.labels[item])

        return outputs

    def __len__(self):
        return len(self.examples)


label_name_index = {'HAND': 0,
                    'KNEE': 1,
                    'EYE': 2,
                    'ANKLE': 3,
                    'SHOULDER': 4,
                    'BACK': 5,
                    'OTHER': 6
                    }

label_index_name = {v: k for k, v in label_name_index.items()}
target_domain_data_path = "/home/yanyandong/Code/pythonProject/data/interaction.labeled.csv"

if __name__ == "__main__":
    # Load dataset
    target_domain_data = pd.read_csv(target_domain_data_path)
    example_texts = target_domain_data['DESC'].tolist()
    example_texts = [utils_preprocess_text(i) for i in example_texts]
    reed_labels = target_domain_data['Label (R Reed)'].tolist()
    brown_labels = target_domain_data['Label (L Brown)'].tolist()
    glenna_labels = target_domain_data['Label (Glenna)'].tolist()
    all_expert_labels = {'expert_1': reed_labels,
                         'expert_2': brown_labels,
                         'expert_3': glenna_labels}

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    dataset = SIRDataset(examples=example_texts,
                         pretrained_tokenizer=tokenizer,
                         max_length=512)

    # Load model
    model = BertClassifier()
    model.load_state_dict(torch.load('/home/yanyandong/Code/pythonProject/trained_models/BERT/bert_model_1'))
    training_args = TrainingArguments(output_dir="/home/yanyandong/Code/pythonProject/outputs",
                                      per_device_eval_batch_size=50,
                                      seed=42,
                                      disable_tqdm=False)
    trainer = Trainer(model=model, args=training_args)

    # Get predicted labels
    outputs = trainer.predict(test_dataset=dataset)
    pred_logits = outputs.predictions
    model_pred_probs = softmax(pred_logits, axis=1)
    preds_top_4 = np.argsort(-model_pred_probs, axis=1)[:, :4]
    model_preds = [label_index_name[i].lower() for i in np.argmax(pred_logits, axis=1)]

    plt.title(f'Micro Avg. ROC')
    colors = ['seagreen', 'seagreen', 'red', 'red', 'dodgerblue', 'dodgerblue']
    color_idx = 0
    for expert in ['expert_1', 'expert_2', 'expert_3']:
        expert_labels = all_expert_labels[expert]
        expert_labels_one_hot = []
        two_expert_labels = []
        two_expert_labels_one_hot = []
        three_expert_labels = []
        three_expert_labels_one_hot = []
        first_expert_label = []
        for label in expert_labels:
            first_expert_label.append(label.split('.')[0].lower())
            two_expert_labels.append(label.lower().split('.')[:2])
            three_expert_labels.append(label.lower().split('.')[:3])
            one_hot_label = [0] * len(label_name_index)
            for sub_label in filter(len, label.split('.')):
                index_ = label_name_index[sub_label.upper()]
                one_hot_label[index_] = 1
            expert_labels_one_hot.append(one_hot_label)
        expert_labels_one_hot = np.array(expert_labels_one_hot)
        for label in two_expert_labels:
            one_hot_label = [0] * len(label_name_index)
            for sub_label in filter(len, label[:2]):
                index_ = label_name_index[sub_label.upper()]
                one_hot_label[index_] = 1
            two_expert_labels_one_hot.append(one_hot_label)
        two_expert_labels_one_hot = np.array(two_expert_labels_one_hot)
        for label in three_expert_labels:
            one_hot_label = [0] * len(label_index_name)
            for sub_label in filter(len, label[:3]):
                index_ = label_name_index[sub_label.upper()]
                one_hot_label[index_] = 1
            three_expert_labels_one_hot.append(one_hot_label)
        three_expert_labels_one_hot = np.array(three_expert_labels_one_hot)

        print(
            f"{expert} F1 replicated (k = 1) = {f1_score(y_true=first_expert_label, y_pred=model_preds, average='micro')}")

        # print('Classification report (k = 1):')
        # print(classification_report(y_true=first_expert_label, y_pred=model_preds,
        #                             target_names=label_name_index, digits=3))

        # convert the pred top 4 to one hot
        preds_top_4_one_hot = []
        for label in preds_top_4:
            one_hot_label = [0] * len(label_name_index)
            for sub_label in label:
                one_hot_label[sub_label] = 1
            preds_top_4_one_hot.append(one_hot_label)

        # Evaluate the F1 score
        print(
            f"{expert} F1 replicated (k = 4) = {f1_score(y_true=expert_labels_one_hot, y_pred=preds_top_4_one_hot, average='micro')}")

        print('Classification report (k = 4):')
        print(classification_report(y_true=expert_labels_one_hot, y_pred=preds_top_4_one_hot,
                                    target_names=label_name_index, digits=3))

        # binarize the labels and plot each of the class
        first_expert_label_one_hot = label_binarize([label_name_index[i.upper()] for i in first_expert_label],
                                                    classes=[i for i in range(len(label_name_index))])

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

        k4_mirco_fpr, k4_mirco_tpr, _ = roc_curve(expert_labels_one_hot.ravel(),
                                                  model_pred_probs.ravel())
        k4_roc_auc = auc(k4_mirco_fpr, k4_mirco_tpr)
        plt.plot(k4_mirco_fpr,
                 k4_mirco_tpr,
                 color=colors[color_idx],
                 lw=2,
                 label=f'{expert} k = 4 (area = {k4_roc_auc:0.2f})')

        # k2_mirco_fpr, k2_mirco_tpr, _ = roc_curve(two_expert_labels_one_hot.ravel(),
        #                                           model_pred_probs.ravel())
        # k2_roc_auc = auc(k2_mirco_fpr, k2_mirco_tpr)
        # plt.plot(k2_mirco_fpr,
        #          k2_mirco_tpr,
        #          color=colors[color_idx],
        #          lw=2,
        #          label=f'{expert} k = 2 (area = {k2_roc_auc:0.2f})')

        # k3_mirco_fpr, k3_mirco_tpr, _ = roc_curve(three_expert_labels_one_hot.ravel(),
        #                                           model_pred_probs.ravel())
        # k3_roc_auc = auc(k3_mirco_fpr, k3_mirco_tpr)
        # plt.plot(k3_mirco_fpr,
        #          k3_mirco_tpr,
        #          color=colors[color_idx],
        #          lw=2,
        #          label=f'{expert} k = 3 (area = {k3_roc_auc:0.2f})')


        color_idx += 1
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'BERT Avg. ROC Curve')
    plt.legend(loc="lower right")
    # plt.show()
    # plt.savefig(f'/home/yanyandong/Code/pythonProject/roc_updated/BERT.png', dpi=300)
    # plt.clf()
