import pickle
from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))
import torch
import onconet.datasets.factory as dataset_factory
import onconet.models.factory as model_factory
from onconet.learn import train
import onconet.transformers.factory as transformer_factory
import onconet.visualize as visualize
import onconet.utils.parsing as parsing
import warnings
import onconet.learn.state_keeper as state
from onconet.utils.get_dataset_stats import get_dataset_stats
import onconet.utils.stats as stats
import csv

DATE_FORMAT_STR = "%Y-%m-%d:%H-%M-%S"


def load_calibrator(path):
    """Load a calibrator pickle, handling sklearn version mismatches."""
    try:
        calibrator = pickle.load(open(path, 'rb'))
        _patch_calibrator(calibrator)
        return calibrator
    except ModuleNotFoundError as e:
        print("WARNING: Could not load calibrator from {}".format(path))
        print("  Cause: {}".format(e))
        print("  This usually means the pickle was saved with an older sklearn.")
        print("  Try using Mirai_pred_rf_callibrator_mar12_2022.p instead.")
        return None


def _patch_calibrator(calibrator):
    """Fix sklearn version mismatches in pickled CalibratedClassifierCV objects.

    Handles renames across sklearn versions:
    - 0.23->0.24: _CalibratedClassifier.calibrators -> calibrators_, classes -> classes_
    - 1.2+: CalibratedClassifierCV.base_estimator -> estimator
    """
    items = calibrator.values() if isinstance(calibrator, dict) else calibrator
    for cal in items:
        if not hasattr(cal, 'calibrated_classifiers_'):
            continue
        # base_estimator -> estimator (sklearn 1.2+)
        if not hasattr(cal, 'estimator') and hasattr(cal, 'base_estimator'):
            cal.estimator = cal.base_estimator
        for cc in cal.calibrated_classifiers_:
            if getattr(cc, 'classes', None) is None and hasattr(cc, 'classes_'):
                cc.classes = cc.classes_
            if not hasattr(cc, 'calibrators'):
                cc.calibrators = cc.__dict__.get('calibrators_', [])
            if not hasattr(cc, 'estimator') and hasattr(cc, 'base_estimator'):
                cc.estimator = cc.base_estimator


def run(args):
    """Run the OncoNet pipeline (train/dev/test) with the given args."""
    if args.ignore_warnings:
        warnings.simplefilter('ignore')

    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        commit = repo.head.object
        args.commit = commit.hexsha
        print("OncoNet main running from commit: \n\n{}\n{}author: {}, date: {}".format(
            commit.hexsha, commit.message, commit.author, commit.committed_date))
    except Exception:
        args.commit = "unknown"
        print("OncoNet main running (git info unavailable)")

    if args.get_dataset_stats:
        print("\nComputing image mean and std...")
        args.img_mean, args.img_std = get_dataset_stats(args)
        print('Mean: {}'.format(args.img_mean))
        print('Std: {}'.format(args.img_std))

    print("\nLoading data-augmentation scheme...")
    transformers = transformer_factory.get_transformers(
        args.image_transformers, args.tensor_transformers, args)
    test_transformers = transformer_factory.get_transformers(
        args.test_image_transformers, args.test_tensor_transformers, args)
    # Load dataset and add dataset specific information to args
    print("\nLoading data...")
    train_data, dev_data, test_data = dataset_factory.get_dataset(args, transformers, test_transformers)
    # Load model and add model specific information to args
    if args.snapshot is None:
        model = model_factory.get_model(args)
    else:
        model = model_factory.load_model(args.snapshot, args)
        if args.replace_snapshot_pool:
            non_trained_model = model_factory.get_model(args)
            model._model.pool = non_trained_model._model.pool
            model._model.args = non_trained_model._model.args


    print(model)
    # Load run parameters if resuming that run.
    args.model_path = state.get_model_path(args)
    print('Trained model will be saved to [%s]' % args.model_path)
    if args.resume:
        try:
            state_keeper = state.StateKeeper(args)
            model, optimizer_state, epoch, lr, epoch_stats = state_keeper.load()
            args.optimizer_state = optimizer_state
            args.current_epoch = epoch
            args.lr = lr
            args.epoch_stats = epoch_stats
        except:
            args.optimizer_state = None
            args.current_epoch = None
            args.lr = None
            args.epoch_stats = None
            print("\n Error loading previous state. \n Starting run from scratch.")
    else:
        print("\n Restarting run from scratch.")


    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        if attr not in ['optimizer_state', 'patient_to_partition_dict', 'path_to_hidden_dict', 'exam_to_year_dict', 'exam_to_device_dict']:
            print("\t{}={}".format(attr.upper(), value))

    save_path = args.results_path
    print()
    if args.train:
        epoch_stats, model = train.train_model(train_data, dev_data, model, args)
        args.epoch_stats = epoch_stats

        if args.plot_losses:
            visualize.viz_utils.plot_losses(epoch_stats)
        print("Save train/dev results to {}".format(save_path))
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path, 'wb'))

    print()
    if args.dev:
        print("-------------\nDev")
        args.dev_stats = train.compute_threshold_and_dev_stats(dev_data, model, args)
        print("Save dev results to {}".format(save_path))
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path, 'wb'))

    if args.test:

        print("-------------\nTest")
        args.test_stats = train.eval_model(test_data, model, args)
        print("Save test results to {}".format(save_path))
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path, 'wb'))

    if (args.dev or args.test) and args.prediction_save_path is not None:
        exams, probs = [], []
        if args.dev:
            exams.extend( args.dev_stats['exams'])
            probs.extend( args.dev_stats['probs'])
        if args.test:
            exams.extend( args.test_stats['exams'])
            probs.extend( args.test_stats['probs'])
        legend = ['patient_exam_id']
        calibrator = None
        if args.callibrator_snapshot is not None:
            calibrator = load_calibrator(args.callibrator_snapshot)
        for i in range(args.max_followup):
            legend.append("{}_year_risk".format(i+1))
        export = {}
        with open(args.prediction_save_path,'w') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=legend)
            writer.writeheader()
            for exam, arr in zip(exams, probs):
                export['patient_exam_id'] = exam
                for i in range(args.max_followup):
                    key = "{}_year_risk".format(i+1)
                    raw_val = arr[i]
                    if calibrator is not None:
                        val = calibrator[i].predict_proba([[raw_val]])[0,1]
                    else:
                        val = raw_val
                    export[key] = val
                writer.writerow(export)
        print("Exported predictions to {}".format(args.prediction_save_path))


if __name__ == '__main__':
    args = parsing.parse_args()
    run(args)
