# defense
Research on adversarial defense


# Flags

- hparams: Hyperparameter set to use. All hyperaparams are specified in `hparams/resnet.py` and are objects of the class `HParams`. You can add more to the hyperaparams and specify in the command

- steps: The number of steps to train for. If this is not specified, `steps` will be calculated from the number of epochs that are mentioned in the hparams file

- resume: Load from existing models

- output_dir: The output directory in which the model has to be saved. It's recommended that *you don't* use this option since the model gets saved with an appropriate name in the `runs/` folder (automatically created). In case you need to specify the `output_dir`, don't do dumb shit like `path/.../model-1` -  If you look back later, you won't be able to figure out what the training settings are.

- use_colab: If you want to train using Google Colab, set this to True. Further instructions ahead.

# Training locally

Example: `python3 train.py --hparams resnet18_default`

That's it.

# Training on colab

- `!git clone https://<access_token>@github.com/srk97/defense.git` -> Paste your GitHub access tokens without `<>`. You can get one from developer options in GitHub settings. 

- `!cd defense && git checkout <branch_name>` -> Do this only if you need to run on a branch

- `!python3 defense/train.py --hparams resnet18_default --use_colab True` -> Substitute with the appropriate hparams setting

This will use your google drive to save the model under `runs` directory which can then be shared with others. You will see a prompt asking for a token when you execute the training script on colab. 
