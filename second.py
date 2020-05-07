#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/
#

import os
import typing
import logging
import time
from dataclasses import dataclass

import pandas
import numpy
import keras


@dataclass
class Relation:
    subj: numpy.array
    rel: numpy.array
    obj: numpy.array

    def texts(self):
        """All texts used in the data."""
        return pandas.concat([self.subj, self.rel, self.obj])

    def transform(self, pyfunc):
        self.subj = numpy.array([pyfunc(x) for x in self.subj])
        self.rel = numpy.array([pyfunc(x) for x in self.rel])
        self.obj = numpy.array([pyfunc(x) for x in self.obj])


@dataclass
class Dataset:
    prem: Relation
    hyp: Relation
    labels: []

    def __str__(self):
        return "prem  : " + str(self.prem) + os.linesep + \
               "hyp   : " + str(self.hyp) + os.linesep + \
               "labels: " + str(self.labels) + os.linesep

    def texts(self):
        """All texts used in the dataset."""
        return pandas.concat([self.prem.texts(), self.hyp.texts()])

    def transform(self, pyfunc):
        self.prem.transform(pyfunc)
        self.hyp.transform(pyfunc)


class Options:
    max_len_padding = 8
    embedding_size = 300  # TODO Replace with detected data dimension.
    directory = "./"


def main(args: Options):
    init_logging()
    logging.info("Loading data ...")
    train, dev, test = load_datasets(args)
    logging.info("Tokenizing data ...")
    input_texts = collect_texts_from_datasets(train, dev, test)
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(input_texts)
    # aux.index.table
    lookup_table = tokenizer.word_index
    # Glove embeddings from http://nlp.stanford.edu/data/glove.6B.zip
    logging.info("Loading Glove ...")
    glove = load_glove_for_vocabulary(args, lookup_table, "glove.6B.300d.txt")
    embedding_matrix = create_embedding_matrix(lookup_table, glove)
    logging.info("Converting texts to sequences ...")
    texts_to_sequences(tokenizer, [train, dev, test])
    logging.info("Padding sequences ...")
    # For each entry in Relation we need it to be of fixed size.
    pad_sequence(args, [train, dev, test])
    logging.info("Creating model ...")
    model = create_model(args, embedding_matrix)
    logging.info("Training model ...")
    start = time.time()
    train_model(
        model, test, dev,
        4096,
        64
    )
    end = time.time()
    logging.info("Training model done in %s", str(end - start))
    # 4096, 256 - 15 min
    #   accuracy: 0.9580, val_accuracy: 0.4132, test accuracy:  91.72
    # 4096,  64 - 16 min
    #   accuracy: 0.9556, val_accuracy: 0.4040, test accuracy:  90.24
    logging.info("Evaluation on test data ...")
    loss, accuracy = model.evaluate([
        test.prem.subj,
        test.prem.rel,
        test.prem.obj,
        test.hyp.subj,
        test.hyp.rel,
        test.hyp.obj,
    ], [test.labels])
    print("Loss    : %.4f" % (loss))
    print("Accuracy: %.2f" % (accuracy * 100))
    # model.save(os.path.join(args.directory, "model-4096-64.tf"))


def init_logging(level=logging.DEBUG):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")


def load_datasets(args: Options):
    train = load_csv_file(os.path.join(args.directory, "rel-train.csv"))
    dev = load_csv_file(os.path.join(args.directory, "rel-dev.csv"))
    test = load_csv_file(os.path.join(args.directory, "rel-test.csv"))
    return train, dev, test


def load_csv_file(file) -> Dataset:
    data_csv = pandas.read_csv(file)
    dataset = Dataset(
        Relation(
            data_csv["PremiseSubj"],
            data_csv["PremiseRel"],
            data_csv["PremiseObj"]),
        Relation(
            data_csv["HypothesisSubj"],
            data_csv["HypothesisRel"],
            data_csv["HypothesisObj"]),
            data_csv["GoldLabel"]
    )
    return dataset


def collect_texts_from_datasets(train: Dataset, dev: Dataset, test: Dataset):
    return pandas.concat([train.texts(), dev.texts(), test.texts()])


def load_glove_for_vocabulary(args: Options, vocabulary, file_name):
    # We do not use Pandas here as we do not need to parse all data,
    # and load them in memory, instead we read just what we need.
    result = {}
    file_path = os.path.join(args.directory, file_name)
    with open(file_path, encoding="utf-8") as input_stream:
        for line in input_stream:
            word, values_as_str = line.rstrip().split(" ", maxsplit=1)
            if word not in vocabulary:
                continue
            try:
                values = [float(token) for token in values_as_str.split(" ")]
            except ValueError:
                print("Invalid line for:", word)
                print(values_as_str)
                raise
            result[word] = numpy.array(values)
    return result


def create_embedding_matrix(lookup_table, words_vectors) -> numpy.ndarray:
    dimension = len(next(iter(words_vectors.values())))
    zero_array = numpy.zeros(dimension)
    # We create a matrix where i-the row is a i-the word,
    # but as the first word has index 1 we add a first row with zeros.
    result = numpy.zeros((len(lookup_table) + 1, dimension))
    for (word, index) in lookup_table.items():
        result[index] = words_vectors.get(word, zero_array)
    return result


def texts_to_sequences(
        tokenizer: keras.preprocessing.text.Tokenizer,
        datasets: typing.List[Dataset]):
    for dataset in datasets:
        dataset.prem.subj = tokenizer.texts_to_sequences(dataset.prem.subj)
        dataset.prem.rel = tokenizer.texts_to_sequences(dataset.prem.rel)
        dataset.prem.obj = tokenizer.texts_to_sequences(dataset.prem.obj)
        dataset.hyp.subj = tokenizer.texts_to_sequences(dataset.hyp.subj)
        dataset.hyp.rel = tokenizer.texts_to_sequences(dataset.hyp.rel)
        dataset.hyp.obj = tokenizer.texts_to_sequences(dataset.hyp.obj)


def pad_sequence(args: Options, datasets: typing.List[Dataset]):
    # We could probably use keras.preprocessing.sequence.pad_sequences
    # instead.
    pad_function = lambda t: pad_list_with_zeros(t, args.max_len_padding)
    for dataset in datasets:
        dataset.transform(pad_function)


def pad_list_with_zeros(values, size: int):
    pad_size = size - len(values)
    return numpy.pad(values, (pad_size, 0), "constant")


def create_model(args: Options, embedding_matrix) -> keras.Model:
    # Encoding of a subject of the premise is obtained by
    # summing word embeddings, the result is fed to a dense layer (64 units).
    from keras.engine.input_layer import Input
    from keras.layers.embeddings import Embedding
    from keras.layers import Dense, Lambda

    input_shape = (args.max_len_padding,)

    input_prem_subj = Input(shape=input_shape)
    input_prem_rel = Input(shape=input_shape)
    input_prem_obj = Input(shape=input_shape)

    input_hyp_subj = Input(shape=input_shape)
    input_hyp_rel = Input(shape=input_shape)
    input_hyp_obj = Input(shape=input_shape)

    embedding = Embedding(
        input_dim=embedding_matrix.shape[0],  # Same as len(lookup_table) + 1
        output_dim=args.embedding_size,
        input_length=args.max_len_padding,
        weights=[embedding_matrix],
        trainable=False
    )

    subob = Dense(units=64, activation=keras.activations.relu)
    relational = Dense(units=20, activation=keras.activations.relu)

    # We can not apply sum function directly, instead we need to put
    # it into layer else we get:
    # AttributeError: 'NoneType' object has no attribute '_inbound_nodes'
    sum = Lambda(lambda x: keras.backend.sum(x, axis=2))

    final_rep = keras.layers.concatenate([
        subob(sum(embedding(input_prem_subj))),
        relational(sum(embedding(input_prem_rel))),
        subob(sum(embedding(input_prem_obj))),
        #
        subob(sum(embedding(input_hyp_subj))),
        relational(sum(embedding(input_hyp_rel))),
        subob(sum(embedding(input_hyp_obj)))
    ])

    predictions = Dense(units=1, activation=keras.activations.sigmoid)(
        Dense(units=16, activation=keras.activations.relu)(final_rep)
    )

    model = keras.models.Model(
        inputs=[
            input_prem_subj,
            input_prem_rel,
            input_prem_obj,
            input_hyp_subj,
            input_hyp_rel,
            input_hyp_obj
        ],
        outputs=predictions
    )

    # keras.utils.vis_utils.plot_model(
    #     model, "./model.png", show_shapes=True, show_layer_names=True)

    # model.summary()

    model.compile(
        optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy"])

    return model


def train_model(
        model: keras.Model, train: Dataset, dev: Dataset, epochs, batch_size):
    model.fit(
        [
            train.prem.subj,
            train.prem.rel,
            train.prem.obj,
            train.hyp.subj,
            train.hyp.rel,
            train.hyp.obj,
        ],
        [train.labels],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=[
            [
                dev.prem.subj,
                dev.prem.rel,
                dev.prem.obj,
                dev.hyp.subj,
                dev.hyp.rel,
                dev.hyp.obj,
            ],
            [dev.labels]
        ]
    )
    # epochs=64  Loss: 0.8598 Accuracy: 60.04
    # epochs=128 Loss: 0.7391 Accuracy: 67.52


if __name__ == "__main__":
    main(Options())
