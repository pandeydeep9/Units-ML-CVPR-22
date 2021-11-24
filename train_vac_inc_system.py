from data_units import MetaLearningSystemDataLoader
from experiment_builder_units import ExperimentBuilder
from few_shot_learning_system_units import MAMLFewShotClassifier
from utils.parser_utils_units import get_args
from utils.dataset_tools import maybe_unzip_dataset

#
# import torch
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
args, device = get_args()

model = MAMLFewShotClassifier(args=args, device=device,
                              im_shape=(2, args.image_channels,
                                        args.image_height, args.image_width))
maybe_unzip_dataset(args=args)
data = MetaLearningSystemDataLoader

maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
maml_system.run_experiment()
