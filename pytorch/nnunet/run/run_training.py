#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import sys
import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration
from nnunet.paths import default_plans_identifier
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")
    parser.add_argument("task", help="can be task name or task id")
    parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-w", required=False, default=None, help="Load pre-trained Models Genesis")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("--continue_from", "--resume_from", dest="continue_from", required=False,
                        choices=("auto", "latest", "best", "final"), default=None,
                        help="checkpoint to resume when continuing training. auto keeps the legacy priority "
                             "final -> latest -> best")
    parser.add_argument("--continue_from_checkpoint", "--resume_from_checkpoint",
                        dest="continue_from_checkpoint", required=False, default=None,
                        help="path to a .model checkpoint to resume from. This implies --continue_training")
    parser.add_argument("--early_stopping_patience", required=False, default=None, type=int,
                        help="enable validation-based early stopping with this patience in epochs")
    parser.add_argument("--early_stopping_start_epoch", required=False, default=0, type=int,
                        help="epoch from which early stopping is allowed to stop training")
    parser.add_argument("--early_stopping_min_delta", required=False, default=0.0, type=float,
                        help="minimum improvement in the validation moving average required to reset early stopping")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False)
    parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the validation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--find_lr", required=False, default=False, action="store_true",
                        help="not used here, just for fun")
    parser.add_argument("--valbest", required=False, default=False, action="store_true",
                        help="hands off. This is not intended to be used")
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
    # parser.add_argument("--interp_order", required=False, default=3, type=int,
    #                     help="order of interpolation for segmentations. Testing purpose only. Hands off")
    # parser.add_argument("--interp_order_z", required=False, default=0, type=int,
    #                     help="order of interpolation along z if z is resampled separately. Testing purpose only. "
    #                          "Hands off")
    # parser.add_argument("--force_separate_z", required=False, default="None", type=str,
    #                     help="force_separate_z resampling. Can be None, True or False. Testing purpose only. Hands off")

    args = parser.parse_args()

    task = args.task
    fold = args.fold
    network = args.network
    network_trainer = args.network_trainer
    weights = args.w
    validation_only = args.validation_only
    plans_identifier = args.p
    find_lr = args.find_lr

    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data

    deterministic = args.deterministic
    valbest = args.valbest
    continue_from = args.continue_from
    continue_from_checkpoint = args.continue_from_checkpoint

    fp32 = args.fp32
    run_mixed_precision = not fp32

    val_folder = args.val_folder
    # interp_order = args.interp_order
    # interp_order_z = args.interp_order_z
    # force_separate_z = args.force_separate_z

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    # if force_separate_z == "None":
    #     force_separate_z = None
    # elif force_separate_z == "False":
    #     force_separate_z = False
    # elif force_separate_z == "True":
    #     force_separate_z = True
    # else:
    #     raise ValueError("force_separate_z must be None, True or False. Given: %s" % force_separate_z)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
    trainer_class, domain = get_default_configuration(network, task, network_trainer, plans_identifier)

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")

    if network == "3d_cascade_fullres":
        assert issubclass(trainer_class, (nnUNetTrainerCascadeFullRes, nnUNetTrainerV2CascadeFullRes)), \
            "If running 3d_cascade_fullres then your " \
            "trainer class must be derived from " \
            "nnUNetTrainerCascadeFullRes"
    else:
        assert issubclass(trainer_class,
                          nnUNetTrainer), "network_trainer was found but is not derived from nnUNetTrainer"

    trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                            deterministic=deterministic,
                            fp16=run_mixed_precision)

    trainer.initialize(not validation_only)

    if args.early_stopping_patience is not None:
        if args.early_stopping_patience < 1:
            raise ValueError("--early_stopping_patience must be >= 1")
        if args.early_stopping_start_epoch < 0:
            raise ValueError("--early_stopping_start_epoch must be >= 0")
        trainer.use_early_stopping = True
        trainer.early_stopping_patience = args.early_stopping_patience
        trainer.early_stopping_start_epoch = args.early_stopping_start_epoch
        trainer.early_stopping_min_delta = args.early_stopping_min_delta
        trainer.print_to_log_file("validation early stopping enabled: patience=%d, start_epoch=%d, min_delta=%g" %
                                  (trainer.early_stopping_patience, trainer.early_stopping_start_epoch,
                                   trainer.early_stopping_min_delta))
    
    if weights != None:                                                         
        trainer.load_pretrained_encoder_weights(weights)
    sys.stdout.flush()

    if find_lr:
        trainer.find_lr()
    else:
        if not validation_only:
            should_continue_training = args.continue_training or continue_from is not None or \
                                       continue_from_checkpoint is not None
            if should_continue_training:
                if continue_from_checkpoint is not None:
                    if not isfile(continue_from_checkpoint):
                        raise RuntimeError("checkpoint not found: %s" % continue_from_checkpoint)
                    trainer.load_checkpoint(continue_from_checkpoint)
                else:
                    if continue_from is None or continue_from == "auto":
                        existing_checkpoints = [
                            join(trainer.output_folder, "model_final_checkpoint.model"),
                            join(trainer.output_folder, "model_latest.model"),
                            join(trainer.output_folder, "model_best.model"),
                        ]
                        if any(isfile(i) for i in existing_checkpoints):
                            trainer.load_latest_checkpoint()
                        else:
                            trainer.print_to_log_file("WARNING! --continue_training was requested, but no checkpoint "
                                                      "was found. Starting training from scratch.")
                    else:
                        checkpoint_files = {
                            "latest": "model_latest.model",
                            "best": "model_best.model",
                            "final": "model_final_checkpoint.model",
                        }
                        checkpoint = join(trainer.output_folder, checkpoint_files[continue_from])
                        if not isfile(checkpoint):
                            if continue_from == "latest":
                                trainer.print_to_log_file("WARNING! latest checkpoint not found: %s. Starting "
                                                          "training from scratch." % checkpoint)
                            else:
                                raise RuntimeError("%s checkpoint not found: %s" % (continue_from, checkpoint))
                        else:
                            trainer.load_checkpoint(checkpoint)
            trainer.run_training()
        else:
            if valbest:
                trainer.load_best_checkpoint(train=False)
            else:
                trainer.load_latest_checkpoint(train=False)

        trainer.network.eval()

        # predict validation
        trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder)

        if network == '3d_lowres':
            trainer.load_best_checkpoint(False)
            print("predicting segmentations for the next stage of the cascade")
            predict_next_stage(trainer, join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1))


if __name__ == "__main__":
    main()
