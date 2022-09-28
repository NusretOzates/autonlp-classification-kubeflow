"""
This is where magic happens. We define the pipeline using the @dsl.pipeline decorator.
The pipeline is composed of 6 steps:

1. Download the dataset
2. Preprocess the dataset
3. Split the dataset
4. Train the model(s)
5. Evaluate the model(s)
6. Print the best model
"""

from typing import NamedTuple

import kfp
import kfp_server_api
from kfp.v2 import dsl
from kfp.v2.dsl import component
from kfp.v2.dsl import Input, Output, Dataset, Metrics
from kfp.compiler.compiler import Compiler


@component(
    packages_to_install=["datasets"],
    output_component_file="upload_data_component.yaml",
)
def upload_data(
        dataset_name: str,
        dataset_subset: str,
        dataset_object: Output[Dataset],
) -> None:
    """Uploads the dataset and preprocesses it

    It will download the dataset from Huggingface.
    Then it will save the dataset inside Kubeflow's minIO object storage.

    Args:
        dataset_name: Name of the dataset from Huggingface
        dataset_subset: Name of the subset if any
        dataset_object: Output dataset object

    Returns:
        Nothing
    """
    from datasets import load_dataset, DatasetDict

    if dataset_subset:
        dataset: DatasetDict = load_dataset(dataset_name, dataset_subset)
    else:
        dataset: DatasetDict = load_dataset(dataset_name)

    dataset.save_to_disk(dataset_object.path)


@component(
    packages_to_install=["datasets", "transformers"],
    output_component_file="upload_data_component.yaml",
)
def preprocess(
        model_name: str,
        dataset_object: Input[Dataset],
        tokenized_dataset_object: Output[Dataset],
) -> None:
    """Preprocess and save the tokenized dataset

    The preprocessing is:

    - Tokenization
    - Padding
    - Truncation
    - Changing the label column to labels

    Args:
        model_name: Name of the model from Huggingface
        dataset_object: Uploaded dataset object
        tokenized_dataset_object: Output tokenized dataset object

    Returns:
        Nothing
    """

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    dataset = load_from_disk(dataset_object.path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)

    dataset = dataset.map(
        function=lambda examples: tokenizer(
            examples["text"], truncation=True, padding="max_length"
        ),
        batched=True,
    )

    dataset.save_to_disk(tokenized_dataset_object.path)


@component(
    packages_to_install=["datasets", "transformers"],
    output_component_file="split_data_component.yaml",
)
def split_data(
        dataset_object: Input[Dataset],
        training_dataset: Output[Dataset],
        validation_dataset: Output[Dataset],
        test_dataset: Output[Dataset],
) -> None:
    """Splits the dataset into training, validation and test

    Split the dataset into training, validation and test.
    The split is 80%, 10% and 10% respectively.
    If the dataset is already split, it will use the split.

    Args:
        dataset_object: Input dataset object
        training_dataset: Output training dataset object
        validation_dataset: Output validation dataset object
        test_dataset: Output test dataset object

    Returns:
        Nothing
    """
    from datasets import load_from_disk, DatasetDict
    from datasets import Dataset as HFDataset

    dataset: DatasetDict = load_from_disk(dataset_object.path)

    splits = len(dataset)

    # We have only the train split
    if splits == 1:
        dataset: DatasetDict = dataset["train"].train_test_split(
            test_size=0.2, stratify_by_column="labels"
        )

        train: Dataset = dataset["train"]
        val = dataset["test"].train_test_split(
            test_size=0.5, stratify_by_column="labels"
        )
        val, test = val["train"], val["test"]

    # We have train and validation (maybe test) split
    if splits == 2:
        split_name = "test" if "test" in dataset else "validation"

        train: HFDataset = dataset["train"]
        val = dataset[split_name].train_test_split(
            test_size=0.5, stratify_by_column="labels"
        )
        val, test = val["train"], val["test"]

    # We have train, validation and test split
    if splits == 3:
        train, val, test = dataset["train"], dataset["validation"], dataset["test"]

    if not train or not val or not test:
        raise ValueError("Something went wrong while splitting the dataset")

    train.save_to_disk(training_dataset.path)
    val.save_to_disk(validation_dataset.path)
    test.save_to_disk(test_dataset.path)


@component(
    packages_to_install=[
        "datasets",
        "transformers",
        "tensorflow",
        "keras-tuner",
        "scipy",
    ],
    output_component_file="training_component.yaml",
)
def model_training(
        model_name: str,
        best_hyperparams: dict,
        training_dataset: Input[Dataset],
        validation_dataset: Input[Dataset],
        test_dataset: Input[Dataset],
        metric: Output[Metrics],
) -> NamedTuple("Output", [("accuracy", float), ("loss", float), ("best_hp", dict)]):
    """Trains the given model with the dataset

    Trains the model with the dataset, if best_hyperparams is an empty dict,
    this component will do hyperparameter tuning and return the best hyperparameters.
    Otherwise, it will concatenate train and validation data and train
    the model.

    Lastly, it will evaluate the data with the test set and return the accuracy and loss.

    Args:
        model_name: Name of the model from Huggingface
        best_hyperparams: the best hyperparameter values. If empty, it will do hyperparameter tuning
        training_dataset: Input training dataset object
        validation_dataset: Input validation dataset object
        test_dataset: Input test dataset object
        metric: Output metric object

    Returns:
        NamedTuple with the accuracy, loss and best hyperparameters
    """

    from typing import NamedTuple

    from datasets import load_from_disk
    from datasets import Dataset as HFDataset
    from transformers import AutoTokenizer, TFAutoModel, DataCollatorWithPadding
    from transformers.modeling_tf_outputs import TFBaseModelOutput

    import tensorflow as tf

    import keras
    from keras import Model
    from keras import Input as KerasInput
    from keras_tuner import HyperParameters, BayesianOptimization
    from keras.layers import Dense

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train: HFDataset = load_from_disk(training_dataset.path)
    val: HFDataset = load_from_disk(validation_dataset.path)
    test: HFDataset = load_from_disk(test_dataset.path)

    num_labels: int = val.features["label"].num_classes

    # For now, this real training code is not activated because it uses too many resources
    def model_builder(hp: HyperParameters) -> Model:

        input_ids = Input(
            name="input_ids",
            shape=tokenizer.init_kwargs["model_max_length"],
            dtype="int32",
        )
        attention_mask = Input(
            name="attention_mask",
            shape=tokenizer.init_kwargs["model_max_length"],
            dtype="int32",
        )

        # I could just use num_labels but I want to use more dense layers
        base_model_output: TFBaseModelOutput = TFAutoModel.from_pretrained(model_name)(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )

        last_hidden_state = base_model_output.last_hidden_state

        # Min is 1 because I don't want to deal with 0. Otherwise, I had to open an extra case for 0
        dense_count = hp.Int(name="Dense Layer Count", min_value=1, max_value=3)

        for i in range(dense_count):
            neuron_count = hp.Int(
                name=f"{i}. Dense Neuron Count", min_value=8, max_value=256
            )

            if i == 0:
                hidden_layer = Dense(neuron_count, activation="relu")(
                    last_hidden_state[:, :, 0]
                )
                continue

            hidden_layer = Dense(neuron_count, activation="relu")(hidden_layer)

        classification_layer = Dense(num_labels, "softmax")(hidden_layer)

        model = Model(
            inputs=[input_ids, attention_mask], outputs=[classification_layer]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def model_builder_small(hp: HyperParameters) -> Model:
        input_ids = KerasInput(
            name="input_ids",
            shape=tokenizer.init_kwargs["model_max_length"],
            dtype="int32",
        )
        attention_mask = KerasInput(
            name="attention_mask",
            shape=tokenizer.init_kwargs["model_max_length"],
            dtype="int32",
        )

        embedding = keras.layers.Embedding(input_dim=40000, output_dim=64)(input_ids)
        sentence_embedding = keras.layers.GlobalAveragePooling1D()(embedding)

        dense_count = hp.Int(name="Dense Layer Count", min_value=1, max_value=3)

        for i in range(dense_count):
            neuron_count = hp.Int(
                name=f"{i}. Dense Neuron Count", min_value=8, max_value=256
            )

            if i == 0:
                hidden_layer = Dense(neuron_count, activation="relu")(
                    sentence_embedding
                )
                continue

            hidden_layer = Dense(neuron_count, activation="relu")(hidden_layer)

        classification_layer = Dense(num_labels, "softmax")(hidden_layer)

        model = Model(
            inputs=[input_ids, attention_mask], outputs=[classification_layer]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def hf_to_tf(dataset: HFDataset) -> tf.data.Dataset:
        """Converts HuggingFace Dataset object into a TF Dataset.

        Args:
            dataset:  HuggingFace Dataset object

        Returns:
            TF Dataset object
        """

        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer, return_tensors="tf"
        )

        return dataset.to_tf_dataset(
            columns=["input_ids", "attention_mask"],
            label_cols=["labels"],
            batch_size=32,
            collate_fn=data_collator,
            shuffle=True,
        )

    tf_train = hf_to_tf(train)
    tf_val = hf_to_tf(val)
    tf_test = hf_to_tf(test)

    OutputTuple = NamedTuple(
        "Output", [("accuracy", float), ("loss", float), ("best_hp", dict)]
    )

    if not best_hyperparams:

        tuner = BayesianOptimization(
            model_builder_small, objective="val_accuracy", max_trials=15
        )

        tuner.search(tf_train, epochs=5, validation_data=tf_val)
        best_hp: HyperParameters = tuner.get_best_hyperparameters()[0]

        for key, value in best_hp.values.items():
            metric.log_metric(key, value)

        """
        We cannot send 'HyperParameters' object to the next component so we need to convert it to a dictionary
        The reason is, we cannot import this object in the type hints.... 
        But this object has get_config() and from_config() methods
        """

        return OutputTuple(0.0, 0.0, best_hp.get_config())

    model = model_builder_small(HyperParameters.from_config(best_hyperparams))

    train_dataset = tf_train.concatenate(tf_val)
    model.fit(train_dataset, epochs=5)

    result = model.evaluate(tf_test, return_dict=True)

    for key, value in result.items():
        metric.log_metric(key, value)

    return OutputTuple(result["accuracy"], result["loss"], {})


@dsl.component(output_component_file="print_best.yaml")
def print_best(all_results: list, metrics: Output[Metrics]) -> None:
    """Prints the best accuracy

    Yes, literally just prints the best accuracy. You can do whatever you want here. Change the code a little,
    and you can send an email, create a Slack message, etc. or upload the best model to a model registry.

    Args:
        best_accuracy: Best accuracy

    Returns:
        None
    """
    all_results.sort(key=lambda x: x.outputs["accuracy"], reverse=True)
    metrics.log_metric("best_accuracy", all_results[0].outputs["accuracy"])


@dsl.pipeline(
    name="text-classification",
)
def classification_training_pipeline(
        dataset_name: str, dataset_subset: str
) -> None:
    """Pipeline for training a text classification model

    Args:
        dataset_name: Name of the dataset
        dataset_subset: Subset of the dataset
        model_names: Name of the models, separated by commas

    Returns:
        None
    """

    upload_op = upload_data(dataset_name, dataset_subset)

    all_results = []

    with dsl.ParallelFor(["?model_names?"]) as model_name:
        dataset_path = upload_op.outputs["dataset_object"]
        preprocess_op = preprocess(model_name, dataset_path)

        tokenized_dataset_path = preprocess_op.outputs["tokenized_dataset_object"]
        split_op = split_data(tokenized_dataset_path)

        training_dataset = split_op.outputs["training_dataset"]
        validation_dataset = split_op.outputs["validation_dataset"]
        test_dataset = split_op.outputs["test_dataset"]
        hp_tune_op = model_training(
            model_name=model_name,
            best_hyperparams={},
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
        )

        best_hp: dict = hp_tune_op.outputs["best_hp"]
        training_op = model_training(
            model_name=model_name,
            best_hyperparams=best_hp,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
        )

        all_results.append(training_op)


    print_best(all_results)



