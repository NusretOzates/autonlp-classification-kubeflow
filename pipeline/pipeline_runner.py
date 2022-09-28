import kfp_server_api
import kfp
from kfp.compiler.compiler import Compiler
import os
from importlib import invalidate_caches


def run_pipeline(
        dataset_name: str = "tweet_eval",
        dataset_subset: str = "emotion",
        model_names: str = "google/electra-small-discriminator",
) -> kfp_server_api.ApiRun:
    """Runs the pipeline

    Args:
        dataset_name: Name of the dataset
        dataset_subset: Subset of the dataset
        model_name: Name of the model

    Returns:
        None
    """

    # Connect to KFP, this command is used to connect to the KFP UI:
    # kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
    client = kfp.Client(host="http://127.0.0.1:8080/pipeline")

    models = model_names.split(",")
    models = [f'\"{model}\"' for model in models]
    model_names = ",".join(models)

    # ParallerFor takes static arguments, it is not possible to give it as pipeline parameter so we need to override the
    # pipeline function code to pass the arguments.

    # Open the pipeline code
    with open("pipeline/pipeline.py", "r") as f:
        pipeline_yaml = f.read()

    # Replace the template with the actual arguments
    pipeline_yaml = pipeline_yaml.replace('"?model_names?"', model_names)

    # Save the pipeline code
    with open("pipeline/pipeline_modified.py", "w") as f:
        f.write(pipeline_yaml)

    # Invalidate the cache to make sure the modified pipeline is loaded
    invalidate_caches()

    # Import the pipeline function
    from pipeline.pipeline_modified import classification_training_pipeline

    # Compile the pipeline
    Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(pipeline_func=classification_training_pipeline, package_path="pipeline.yaml")

    # Create an experiment
    experiment = client.create_experiment(name="text-classification", description="Text classification pipeline")

    # Run the pipeline
    job = client.run_pipeline(
        experiment_id=experiment.id,
        job_name="text-classification",
        pipeline_package_path="pipeline.yaml",
        enable_caching=False,
        params={
            "dataset_name": dataset_name,
            "dataset_subset": dataset_subset
        },
    )

    # Delete the modified pipeline code
    os.remove("pipeline/pipeline_modified.py")

    components = filter(lambda x: x.endswith(".yaml"), os.listdir())

    for file in components:
        os.remove(file)

    return job
