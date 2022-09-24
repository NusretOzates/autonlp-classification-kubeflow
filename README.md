# AutoNLP Text Classification Tool

This is a simple tool to train and deploy text classification models. It is built on top of [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
Tensorflow models. It uses [Keras-Tuner](https://keras.io/keras_tuner/) to find the best hyperparameters for the model. It uses [Kubeflow](https://www.kubeflow.org/) Pipelines to train the model
and you can use this in your local machine using [minikube](https://minikube.sigs.k8s.io/docs/) or in a cloud environment like GCP or AWS.

To get user input, it uses [Streamlit](https://streamlit.io/). Currently, there is no serving or deployment of the model.
But if you use this in a cloud environment, you can use this project to train the model and then use the model in your
production environment. You only need to add a new component to the pipeline to deploy the model. I can add that component
in the future as there is already a [Kubeflow component](https://github.com/kubeflow/pipelines/tree/master/components/contrib/google-cloud/ml_engine/deploy) for it.

## Used Technologies

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [Keras-Tuner](https://keras.io/keras_tuner/)
- [Kubeflow](https://www.kubeflow.org/)
- [Streamlit](https://streamlit.io/)
- [minikube](https://minikube.sigs.k8s.io/docs/)


## How to use

### Local

- Install [minikube](https://minikube.sigs.k8s.io/docs/) and start it with `minikube start` command. Don't forget to add some resources to it. For example: `minikube start --cpus 8 --memory 12g --disk-size 60g`
- Install [kubeflow](https://www.kubeflow.org/docs/components/pipelines/v1/installation/localcluster-deployment/#deploying-kubeflow-pipelines) and connect it with `kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80` command.
- Install the requirements with `pip install -r requirements.txt` command.
- Run the app with `streamlit run Trainer.py` command.
- You are ready to go!

### Cloud

- Install Kubeflow to your cloud environment's Kubernetes cluster.
- Install the requirements with `pip install -r requirements.txt` command.
- Change the Kubeflow host in `pipeline/pipeline.py` file.
- Run the app with `streamlit run Trainer.py` command.
- As you connect your cloud environment's cluster, you will most probably have necessary permissions to access cloud resources such as GC Storage, AWS S3, etc.