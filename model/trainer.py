import sys
import csv
import math
import spacy
from spacy.tokens import DocBin
from spacy.training.initialize import init_nlp
from spacy.training.loop import train
from spacy.training import Example
from random import shuffle
from pathlib import Path

CONFIG_PATH = Path("config.cfg")
TRAIN_DATA_PATH = Path("train.spacy")
OUTPUT_PATH = Path("./output")

def process_data() -> list:
    data = []
    with open("synthetic_data.csv", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
        for row in reader:
            data.append((row[0].lower(), row[1]))

    return data


def test_model(nlp, test_data: list):
    correct = 0
    total = len(test_data)

    for text, true_label in test_data:
        doc = nlp(text)
        predicted_label = max(doc.cats, key=doc.cats.get)
        # print(
        #     f"Predicted: {predicted_label}, True Label: {true_label}, Probabilities: {doc.cats}"
        # )
        if predicted_label == true_label:
            correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2f}")


def get_training_docbin(data, nlp):
    doc_bin = DocBin()
    for text, label in data:
        doc = nlp.make_doc(text)
        doc.cats = {
            "bank_statement": label == "bank_statement",
            "invoice": label == "invoice",
            "drivers_licence": label == "drivers_licence",
        }
        doc_bin.add(doc)
    return doc_bin


def train_model(config_path, train_data_path, output_path, epochs: int = 3):
    config = spacy.util.load_config(config_path)
    config["paths"]["train"] = str(train_data_path)
    config["paths"]["dev"] = str(train_data_path)
    config["training"]["max_epochs"] = epochs
    nlp = init_nlp(config)

    train(nlp)
    nlp.to_disk(output_path)
    print("Model trained and saved")


def get_train_test_split() -> tuple[list, list]:
    all_data = process_data()
    data_length = len(all_data)
    train_data = all_data[: math.floor(data_length * 0.8)]
    print("Train data length: ", len(train_data))
    test_data = all_data[math.floor(data_length * 0.8) :]
    print("Test data length: ", len(test_data))

    return train_data, test_data


if __name__ == "__main__":
    train_data, test_data = get_train_test_split()
    action = sys.argv[1]
    if action == 'train':
        epochs = int(sys.argv[2])
        nlp = spacy.load("en_core_web_sm")
        train_docbin = get_training_docbin(train_data, nlp)
        train_docbin.to_disk(TRAIN_DATA_PATH)
        train_model(CONFIG_PATH, TRAIN_DATA_PATH, OUTPUT_PATH, epochs=epochs)

#    if action == 'test':
        nlp = spacy.load(OUTPUT_PATH)
        test_model(nlp, test_data)
