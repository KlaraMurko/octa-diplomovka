"""
classifier/scripts
------------------
OCTA Disease Classifier pipeline.
"""

from .dataset import (
    EXPERIMENT_CONFIGS,
    LABEL_NAMES,
    OctaClsDataset,
    build_dataloaders,
    load_dataframes,
)
from .evaluation import (
    compute_min_recall,
    compute_per_class_recall,
    evaluate,
    plot_confusion_matrices,
    print_metrics,
)
from .gradcam import run_gradcam
from .model import (
    FUSION_TYPES,
    OctaClassifier,
    ViTEncoder,
    build_model,
    load_encoder,
)
from .training import (
    TrainConfig,
    load_best_model,
    plot_training_curves,
    run_grid_search,
    save_best_model,
    train_one_run,
)
from .utils import (
    Timer,
    append_to_summary_csv,
    compute_class_weights,
    make_run_dir,
    now_str,
    save_json,
)
