import os
import pathlib
import sys
import tempfile
from functools import partial
from pathlib import Path

import torch
from ray.tune.search.hyperopt import HyperOptSearch
from torch.optim import Adam

from src.experiments.configs.config import BraTS2020Configuration
from src.experiments.training import training, early_stopping, save_learning_metrics, validate_model, train_model, \
    process_augmentations, set_up_recordings, loss_function, set_up_functions, set_up_current_recordings

from ray.air import session, CheckpointConfig
from ray import train, tune
from ray.train import RunConfig, Checkpoint
from ray.tune.schedulers import ASHAScheduler

from src.experiments.utility_functions.data_access import derive_loader
from src.models.UNet.unet2d import UNet


def grid_training(grid_config, constant_param):
    # Set up Model parameters

    classes = constant_param["classes"]
    selected_mri_vols = grid_config["selected_mri"]
    channels = len(selected_mri_vols)
    print(channels)
    print(grid_config["layers"])

    model = UNet(in_channels=channels, classes=classes,
                 layers=grid_config["layers"],
                 dropout_p=grid_config["dropout_rate"])

    # # If GPu is specified
    # gpu = training_config["GPU"]
    # if gpu:
    #     model.to(device="cuda" if torch.cuda.is_available() else "cpu")

    # Set up main training hyperparameters
    loss_func = loss_function(grid_config["loss"])
    lr = grid_config["ilr"]
    weight_decay = grid_config["weight_decay"]
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    clip_value = grid_config["gradient_clipping"]

    # Set up patience and best validation loss
    best_valid = 0
    stagnation = 0
    patience = grid_config["patience"]

    # calculate steps per epoch for training and test set
    learning_metrics = constant_param["learning_metrics"]
    lm_funcs = set_up_functions(learning_metrics)
    training_recordings = set_up_recordings(learning_metrics)
    current_metrics = set_up_current_recordings(learning_metrics)
    validation_recordings = set_up_recordings(learning_metrics)

    # # Get experiment name
    # exp_name = config["experiment"]
    # save_path = config["save_path"]

    # # Save the experiment deep learning model
    # model_output = checkpoint_dir
    # if not os.path.exists(model_output):
    #     os.makedirs(model_output)

    print("[INFO] training the network...")
    epoch = grid_config["epoch"]
    batch = grid_config["batch"]
    verbose_mode = True
    data_dir = constant_param["data_dir"]
    augmentations = process_augmentations(constant_param["augmentations"])
    # Get training

    for e in range(epoch):
        # Load the train loader
        train_loader = derive_loader(data_directory=data_dir, purpose="training",
                                     mri_vols=selected_mri_vols, transforms=augmentations,
                                     batch=batch)

        # Train the model and update the metric recordings
        train_model(train_loader, model, loss_func=loss_func,
                    opt=opt, learning_metrics=current_metrics, training_records=training_recordings,
                    learning_functions=lm_funcs, clip_value=clip_value, device="cpu")

        validate_model(data_dir, model, loss_func=loss_func,
                       validation_recordings=validation_recordings,
                       learning_functions=lm_funcs, batch=batch,
                       mri_vols=selected_mri_vols, device="cpu")

        validation_loss = validation_recordings["loss"][e]

        # Send the current training result back to Tune
        train.report({"loss": validation_loss})

        # # see if early stopping is required
        # torch.save({
        #     'model_state_dict': model.state_dict(),
        #     'optim_state_dict': opt.state_dict(),
        #     'epoch': e,
        # }, os.path.join(model_output, "model.pth"))
        #
        # save_learning_metrics(save_path, exp_name, training_recordings, "training")
        # save_learning_metrics(save_path, exp_name, validation_recordings, "validation")

        # Determine if early stopping is necessary
        best_valid, stagnation = early_stopping(validation_loss=validation_recordings["loss"][e],
                                                best_valid=best_valid, stagnation=stagnation)
        if stagnation >= patience:
            break

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None

            should_checkpoint = epoch % grid_config.get("checkpoint_freq", 1) == 0
            # In standard DDP training, where the model is the same across all ranks,
            # only the global rank 0 worker needs to save and report the checkpoint
            if train.get_context().get_world_rank() == 0 and should_checkpoint:
                torch.save(
                    model.module.state_dict(),  # NOTE: Unwrap the model.
                    os.path.join(temp_checkpoint_dir, "model.pt"),
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            train.report({"loss": validation_loss}, checkpoint=checkpoint)

    # # Save the experiment deep learning model
    # model_output = os.path.join(save_path, exp_name)
    # if not os.path.exists(model_output):
    #     os.makedirs(model_output)
    # torch.save(model, os.path.join(model_output, "model.pth"))
    #
    # # Save the loss and other defined metrics for training and validation
    # save_learning_metrics(save_path, exp_name, training_recordings, "training")
    # save_learning_metrics(save_path, exp_name, validation_recordings, "validation")


if __name__ == "__main__":
    # config = BraTS2020Configuration(sys.argv[1])

    save_path = r"D:\Users\James\Code_Projects\CS4_CompNeuroProj\CS4_CompNeuroProj\src\experiments\saved_model\test_grid_search"
    data_directory = Path(r"D:\Users\James\Code_Projects\CS4_CompNeuroProj\CS4_CompNeuroProj\src\data\processed\test_binary_experiment\standard_test")

    constant_grid = {
        "data_dir": data_directory,
        "classes": 2,
        "learning_metrics": ["accuracy", "hausdorff", "IoU"],
        "augmentations":
            [{"vertical_flipping": 0.5}, {"horizontal_flipping": 0.5}, {"rotation": 90}],
    }


    grid_config = {

        "epoch": tune.grid_search([1]),
        "batch": tune.grid_search([32]),
        "gradient_clipping": tune.grid_search([1, 3, 5]),
        "patience": tune.grid_search([5]),
        # "patience": tune.grid_search([5, 10, 15, 30]),
        "ilr": tune.grid_search([0.0001, 0.001, 0.01]),
        "weight_decay": tune.grid_search([0.0001, 0.001, 0.01]),
        "loss": {
            "loss_coefficients": tune.grid_search([[0.5, 0.5], [1.0, 0], [0, 1.0], [0.7, 0.3], [0.3, 0.7]]),
            "focal_alpha": tune.grid_search([0.5, 1, 1.5, 2, 3]),
            "focal_gamma": tune.grid_search([0.5, 1, 1.5, 2, 3])
        },
        "layers": tune.grid_search([[64, 128, 256], [128, 256, 512]]),
        "dropout_rate": tune.grid_search([0.25, 0.5, 0.75]),
        "selected_mri": tune.grid_search([[1, 2, 3]])
        # "selected_mri": tune.grid_search([[0], [1], [2], [3],  [1, 2, 3], [0, 1, 2, 3]])
    }

    # scheduler = ASHAScheduler(
    #     metric="loss",
    #     mode="min",
    #     max_t=1,
    #     grace_period=1,
    #     reduction_factor=2,
    # )

    # hyperopt_search = HyperOptSearch(grid_config, metric="loss", mode="min")
    run_config = RunConfig(storage_path="D:\\Users\\James\\ray_tune_experiments", name="test_grid_search",
                           checkpoint_config=CheckpointConfig(num_to_keep=1,
                                                              checkpoint_score_attribute="mean_loss",
                                                              checkpoint_score_order="min",))

    total_combinations = 1
    for key, values in grid_config.items():
        if key == "loss":
            for nkey, nvalues in values.items():
                total_combinations *= len(nvalues)
        else:
            total_combinations *= len(values)

    print(total_combinations)

    trainable_with_cpu_gpu = tune.with_resources(tune.with_parameters(grid_training, constant_param=constant_grid),
                                                 {"cpu": 2})

    tuner = tune.Tuner(
        trainable_with_cpu_gpu,
        tune_config=tune.TuneConfig(
            num_samples=total_combinations,
            scheduler=ASHAScheduler(metric="loss", mode="min"),
        ),
        param_space=grid_config,
        run_config=run_config
    )
    results = tuner.fit()

    # save_config(config, training_config=config.training)
