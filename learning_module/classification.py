from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, RobertaForSequenceClassification
import torch
from torch import nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch import cuda
from torch import optim as optim
from sklearn import metrics
import numpy as np
import random
from transformers import AdamW
from transformers import get_scheduler
from utils import prepare_train_data, prepare_test_data
import logging
logging.basicConfig(filename='running.log', level=logging.INFO)
use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

directory = os.path.dirname(os.path.abspath(__file__))

model_folder_path = os.path.join(directory, 'model')
print(model_folder_path)
print(device)
torch.backends.cudnn.benchmark = True
false_cases = []

PATCH_FIRST_PHASE_PARAMS = {'batch_size': 5, 'shuffle': True, 'num_workers': 6}
PATCH_SECOND_PHASE_PARAMS = {'batch_size': 5, 'shuffle': True, 'num_workers': 6}

HIDDEN_DIM = 768

FIRST_PHASE_NUMBER_OF_EPOCHS = 6
SECOND_PHASE_NUMBER_OF_EPOCHS = 200

FIRST_PHASE_LEARNING_RATE = 1e-4
SECOND_PHASE_LEARNING_RATE = 1e-5

NEED_FIRST_PHASE = False

HIDDEN_DIM_DROPOUT_PROB = 0.3
NUMBER_OF_LABELS = 2

class PatchDataset(Dataset):
    def __init__(self, data_type):
        if data_type == "test":
            self.data = prepare_test_data()
        
        elif data_type == "train":
            self.data = prepare_train_data()

        else:
            print("invalid dataset")
            exit()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        patch = self.data[index]
        return patch["buggy"], patch["att_buggy"], patch["fixed"], patch["att_fixed"], torch.tensor(patch["label"]), patch["origin_patch"]

class PatchClassifier(nn.Module):
    def __init__(self):
        super(PatchClassifier, self).__init__()
        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)

        self.l1 = nn.Linear(2*HIDDEN_DIM, 256)
        self.l2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        #16-18Aug's Model
        # self.l1 = nn.Linear(2 * HIDDEN_DIM, 2 * HIDDEN_DIM)
        # self.dense = nn.Linear(2 * HIDDEN_DIM, 2 * HIDDEN_DIM)
        # self.drop_out = nn.Dropout(2 * HIDDEN_DIM_DROPOUT_PROB)
        # self.out_proj = nn.Linear(2 * HIDDEN_DIM, NUMBER_OF_LABELS)

    def forward(self, buggy, fixed, buggy_attention_mask, fixed_attention_mask):
        bug_vec = self.code_bert(input_ids=buggy, attention_mask=buggy_attention_mask)
        bug_out = bug_vec.last_hidden_state[:, 0, :]
        fix_vec = self.code_bert(input_ids=fixed, attention_mask=fixed_attention_mask)
        fix_out = fix_vec.last_hidden_state[:, 0, :]
        final_vec = torch.cat([bug_out, fix_out], axis=1)

        #Night18Aug Model
        x = self.l1(final_vec)
        x = self.relu(x)
        x = self.l2(x)
        #16-18Aug's Model
        # x = self.l1(final_vec)
        # x = self.drop_out(x)
        # x = self.dense(x)
        # x = torch.tanh(x)
        # x = self.drop_out(x)
        # x = self.out_proj(x)

        return x, final_vec

def predict_test_data(model, testing_generator, device):
    # loss_function = nn.NLLLoss()
    print("Testing...")
    model.eval()
    y_pred = []
    y_test = []
    failed_predictions = []
    test_data_length = 0
    with torch.no_grad():
        for buggy_progs, att_buggy, fixed_progs, att_fixed, labels, _ in testing_generator:
            buggy_progs, att_buggy, fixed_progs, att_fixed, labels = buggy_progs.to(device), att_buggy.to(device), fixed_progs.to(device), att_fixed.to(device), labels.to(device)
            outs = model(buggy_progs, fixed_progs, att_buggy, att_fixed)[0]
            outs = F.log_softmax(outs, dim=1)
            # loss = loss_function(outs, labels)
            for i in range(len(labels)):
                probabilities = outs[i]
                label = labels[i].item()
                actual = (probabilities.topk(1))[1].item()
                y_pred.append(actual)
                y_test.append(label)

        confusion_matrix = metrics.confusion_matrix(y_pred=y_pred, y_true=y_test)
        precision = metrics.precision_score(y_pred=y_pred, y_true=y_test)
        recall = metrics.recall_score(y_pred=y_pred, y_true=y_test)
        # f1 = metrics.fbeta_score(y_pred=y_pred, y_true=y_test, average='weighted', beta=0.5)
        f1 = metrics.f1_score(y_pred=y_pred, y_true=y_test)
    model.train()
    print("Finish testing")
    return precision, recall, f1, confusion_matrix

def train(model, need_freeze_param, learning_rate, number_of_epochs, training_generator, testing_generator, is_continue = False):
    loss_function = nn.NLLLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = number_of_epochs * len(training_generator)
    # lr_scheduler = get_scheduler(
    #     "linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=num_training_steps
    # )
    print("Testing before training ...")
    best_result  = {}
    if is_continue:
        precision, recall, f1, confusion_matrix = predict_test_data(model=model,
                                                    testing_generator=testing_generator,
                                                    device=device)
        best_result['f1'] = f1
        best_result['recall'] = recall
        best_result['precision'] = precision
        best_result["cf"] = confusion_matrix
    else:
        best_result['f1'] = 0
        best_result['recall'] = 0
        best_result['precision'] = 0
        best_result["cf"] = np.array([[0, 999], [0, 0]])


    iterations = 0
    for epoch in range(number_of_epochs):
        model.train()
        total_loss = 0
        for buggy_progs, att_buggy, fixed_progs, att_fixed, labels, _ in training_generator:
            iterations += 1
            buggy_progs, att_buggy, fixed_progs, att_fixed, labels = buggy_progs.to(device), att_buggy.to(device), fixed_progs.to(device), att_fixed.to(device), labels.to(device)
            outs = model(buggy_progs, fixed_progs, att_buggy, att_fixed)[0]
            outs = F.log_softmax(outs, dim=1)
            loss = loss_function(outs, labels)
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            model.zero_grad()
            total_loss += loss.detach().item()
            print("Iter {}, Loss {}".format(iterations, loss.detach().item()))
            logging.info("Iter {}, Loss {}".format(iterations, loss.detach().item()))
            if iterations % 50 == 0:
                precision, recall, f1, confusion_matrix = predict_test_data(model=model,
                                                    testing_generator=testing_generator,
                                                    device=device)
                print("Precision: {}".format(precision))
                print("Recall: {}".format(recall))
                print("F1: {}".format(f1))
                print(confusion_matrix)
                logging.info("[TESTING]Precision {}, Recall {}, F1 {}, CF_matrix {}".format(precision, recall, f1, confusion_matrix))
                fp = confusion_matrix[0,1]
                tp = confusion_matrix[1,1]
                if f1 > best_result['f1']:
                # if (fp != 0 and fp < best_result['cf'][0,1]) or (fp == best_result['cf'][0,1] and tp > best_result['cf'][1,1]):
                    torch.save(model.state_dict(), model_folder_path + "/model_best.pth".format(iterations))
                    best_result['f1'] = f1
                    best_result['recall'] = recall
                    best_result['precision'] = precision
                    best_result["cf"] = confusion_matrix

                logging.info("[BEST RESULT]Precision {}, Recall {}, F1 {}, CF_matrix {}".format(best_result["precision"], best_result["recall"], best_result["f1"], best_result["cf"]))
                # model.train()

        logging.info("[AVERAGE] Loss {}".format(total_loss))
        total_loss = 0      
    
    return best_result

def do_train():
    #Load training data
    print("Loading training data")
    training_set = PatchDataset("train")
    first_training_generator = DataLoader(training_set, **PATCH_FIRST_PHASE_PARAMS)
    second_training_generator = DataLoader(training_set, **PATCH_SECOND_PHASE_PARAMS)

    #Load testing data
    print("Loading testing data")
    testing_set = PatchDataset("test")
    first_testing_generator = DataLoader(testing_set, **PATCH_FIRST_PHASE_PARAMS)
    second_testing_generator = DataLoader(testing_set, **PATCH_SECOND_PHASE_PARAMS)

    model = PatchClassifier()
    model.to(device)

    if NEED_FIRST_PHASE:
        print("First training phase...")
        best_result = train(model=model, need_freeze_param=True, learning_rate=FIRST_PHASE_LEARNING_RATE, number_of_epochs=FIRST_PHASE_NUMBER_OF_EPOCHS, training_generator=first_training_generator, testing_generator=first_testing_generator)
        logging.info("[FINAL PHASE1] Precision {}, Recall {}, F1 {}, CF_matrix {}".format(best_result["precision"], best_result["recall"], best_result["f1"], best_result["cf"]))
    print("Second training phase...")
    best_result = train(model=model, need_freeze_param=False, learning_rate=SECOND_PHASE_LEARNING_RATE, number_of_epochs=SECOND_PHASE_NUMBER_OF_EPOCHS, training_generator=second_training_generator, testing_generator=second_testing_generator, is_continue=False)
    logging.info("[FINAL PHASE2] Precision {}, Recall {}, F1 {}, CF_matrix {}".format(best_result["precision"], best_result["recall"], best_result["f1"], best_result["cf"]))

def do_test():
    print("Loading testing data")
    testing_set = PatchDataset("test")
    first_testing_generator = DataLoader(testing_set, **PATCH_FIRST_PHASE_PARAMS)
    second_testing_generator = DataLoader(testing_set, **PATCH_SECOND_PHASE_PARAMS)
    model = PatchClassifier()
    model.to(device)
    model.load_state_dict(torch.load("model/model15Aug.pth"))
    precision, recall, f1, confusion_matrix = predict_test_data(model=model,
                                                        testing_generator=second_testing_generator,
                                                        device=device)
    print("[TESTING] Precision {}, Recall {}, F1 {}, CF_matrix {}".format(precision, recall, f1, confusion_matrix))
                   
def get_confidence_score(path = "cf.txt", model_path = "model/modelbest.pth"):
    f = open(path, "w")
    testing_set = PatchDataset("test")
    testing_generator = DataLoader(testing_set, **PATCH_FIRST_PHASE_PARAMS)
    model = PatchClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    y_pred = []
    y_test = []
    failed_predictions = []
    with torch.no_grad():
        for buggy_progs, att_buggy, fixed_progs, att_fixed, labels, origin_patches in testing_generator:
            buggy_progs, att_buggy, fixed_progs, att_fixed, labels = buggy_progs.to(device), att_buggy.to(device), fixed_progs.to(device), att_fixed.to(device), labels.to(device)
            outs = model(buggy_progs, fixed_progs, att_buggy, att_fixed)[0]
            outs = F.softmax(outs, dim=1)
            for i in range(len(labels)):
                probabilities = outs[i]
                label = labels[i].item()
                actual = 0
                if probabilities[1] > probabilities[0]:
                    actual = 1
                y_pred.append(actual)
                y_test.append(label)
                f.write("{}\t{}\t{}\n".format(origin_patches[i], probabilities[0], probabilities[1]))

        confusion_matrix = metrics.confusion_matrix(y_pred=y_pred, y_true=y_test)
        precision = metrics.precision_score(y_pred=y_pred, y_true=y_test)
        recall = metrics.recall_score(y_pred=y_pred, y_true=y_test)
        f1 = metrics.f1_score(y_pred=y_pred, y_true=y_test)
        print(confusion_matrix)
        print(confusion_matrix[0,1])
    f.close()   


if __name__ == '__main__':
    torch.manual_seed(0)
    # get_confidence_score(path = "cf.txt", model_path = "model/modelbest.pth")
    do_train()
