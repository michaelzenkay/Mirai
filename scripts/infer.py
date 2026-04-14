"""Programmatic Mirai inference helpers used by mg/utils.py."""

import copy
import os
import pickle
import sys
from argparse import Namespace
from glob import glob
from os.path import dirname, join, realpath

_MIRAI_ROOT = dirname(dirname(realpath(__file__)))
if _MIRAI_ROOT not in sys.path:
    sys.path.insert(0, _MIRAI_ROOT)

import pandas as pd
import torch

import onconet.datasets  # noqa: F401
import onconet.models  # noqa: F401
import onconet.transformers  # noqa: F401
import onconet.datasets.factory as dataset_factory
import onconet.models.factory as model_factory
import onconet.transformers.factory as transformer_factory
from onconet.learn import train


EMPTY_NAME_ERR = (
    "Name of transformer or one of its arguments cant be empty\n"
    'Use "name/arg1=value/arg2=value" format'
)
BATCH_SIZE_SPLIT_ERR = "batch_size (={}) should be a multiple of batch_splits (={})"
DATA_AND_MODEL_PARALLEL_ERR = (
    "data_parallel and model_parallel should not be used in conjunction."
)
CONFLICTING_WEIGHTED_SAMPLING_ERR = (
    "Cannot both use class_bal and year_weighted_class_bal at the same time."
)


def _find_file(directory, pattern):
    matches = sorted(glob(join(directory, pattern)))
    if not matches:
        return None
    return matches[0]


def _parse_transformers(raw_transformers):
    transformers = []
    for transformer in raw_transformers:
        arguments = transformer.split("/")
        name = arguments[0]
        if name == "":
            raise ValueError(EMPTY_NAME_ERR)
        kwargs = {}
        for argument in arguments[1:]:
            pieces = argument.split("=")
            var = pieces[0]
            val = pieces[1] if len(pieces) > 1 else None
            if var == "":
                raise ValueError(EMPTY_NAME_ERR)
            kwargs[var] = val
        transformers.append((name, kwargs))
    return transformers


def _parse_block_layout(raw_block_layout):
    block_layout = []
    for raw_layer_layout in raw_block_layout:
        raw_block_specs = raw_layer_layout.split("-")
        layer = [raw_block_spec.split(",") for raw_block_spec in raw_block_specs]
        layer = [(block_name, int(num_repeats)) for block_name, num_repeats in layer]
        block_layout.append(layer)
    return block_layout


def _validate_args(args):
    if args.batch_size % args.batch_splits != 0:
        raise ValueError(BATCH_SIZE_SPLIT_ERR.format(args.batch_size, args.batch_splits))
    if args.data_parallel and args.model_parallel:
        raise ValueError(DATA_AND_MODEL_PARALLEL_ERR)
    if args.class_bal and args.year_weighted_class_bal:
        raise ValueError(CONFLICTING_WEIGHTED_SAMPLING_ERR)
    assert args.ten_fold_test_index in range(-1, 10)


def _patch_calibrator(calibrator):
    items = calibrator.values() if isinstance(calibrator, dict) else calibrator
    for cal in items:
        if not hasattr(cal, "calibrated_classifiers_"):
            continue
        if not hasattr(cal, "estimator") and hasattr(cal, "base_estimator"):
            cal.estimator = cal.base_estimator
        for calibrated in cal.calibrated_classifiers_:
            if getattr(calibrated, "classes", None) is None and hasattr(calibrated, "classes_"):
                calibrated.classes = calibrated.classes_
            if not hasattr(calibrated, "calibrators"):
                calibrated.calibrators = calibrated.__dict__.get("calibrators_", [])
            if not hasattr(calibrated, "estimator") and hasattr(calibrated, "base_estimator"):
                calibrated.estimator = calibrated.base_estimator


def _load_calibrator(path):
    if path is None:
        return None
    try:
        with open(path, "rb") as handle:
            calibrator = pickle.load(handle)
        _patch_calibrator(calibrator)
        return calibrator
    except ModuleNotFoundError as exc:
        print("WARNING: Could not load calibrator from {}".format(path))
        print("  Cause: {}".format(exc))
        print("  This usually means the pickle was saved with an older sklearn.")
        print("  Try using Mirai_pred_rf_callibrator_mar12_2022.p instead.")
        return None


def _build_args(
    metadata_csv,
    model_dir,
    calibrate=True,
    batch_size=1,
    img_mean=7047.99,
    img_std=12005.5,
    img_size=(1664, 2048),
    num_workers=0,
    use_cuda=False,
):
    encoder_path = _find_file(model_dir, "*Base*")
    transformer_path = _find_file(model_dir, "*Transformer*")
    calibrator_path = None
    if calibrate:
        calibrator_path = (
            _find_file(model_dir, "*mar12_2022*callibrator*")
            or _find_file(model_dir, "*callibrator_mar*")
            or _find_file(model_dir, "*callibrator*")
        )

    if encoder_path is None:
        raise FileNotFoundError(
            "No image encoder snapshot (*Base*) found in {}".format(model_dir)
        )
    if transformer_path is None:
        raise FileNotFoundError(
            "No transformer snapshot (*Transformer*) found in {}".format(model_dir)
        )

    args = Namespace(
        train=False,
        test=True,
        dev=False,
        threshold=None,
        ensemble_paths=[],
        train_years=[],
        dev_years=[],
        test_years=[],
        predict_birads=False,
        predict_birads_lambda=0,
        invasive_only=False,
        rebalance_eval_cancers=False,
        downsample_activ=False,
        confidence_interval=0.95,
        num_resamples=10000,
        dataset="csv_mammo_risk_all_full_future",
        image_transformers=[],
        tensor_transformers=[],
        test_image_transformers=[],
        test_tensor_transformers=[],
        num_workers=num_workers,
        img_size=list(img_size),
        patch_size=[-1, -1],
        get_dataset_stats=False,
        get_activs_instead_of_hiddens=False,
        img_mean=[img_mean],
        img_std=[img_std],
        img_dir="",
        num_chan=3,
        force_input_dim=False,
        input_dim=512,
        transfomer_hidden_dim=512,
        num_heads=8,
        multi_image=True,
        num_images=4,
        pred_both_sides=False,
        min_num_images=4,
        video=False,
        metadata_dir=None,
        metadata_path=metadata_csv,
        cache_path=None,
        drop_benign_side=False,
        class_bal=True,
        shift_class_bal_towards_imediate_cancers=False,
        year_weighted_class_bal=False,
        device_class_bal=False,
        allowed_devices="all",
        use_c_view_if_available=False,
        use_spatial_transformer=False,
        spatial_transformer_name="affine",
        spatial_transformer_img_size=[208, 256],
        location_network_name="resnet18",
        location_network_block_layout=[
            "BasicBlock,2",
            "BasicBlock,2",
            "BasicBlock,2",
            "BasicBlock,2",
        ],
        tps_grid_size=10,
        tps_span_range=0.9,
        use_region_annotation=False,
        fraction_region_annotation_to_use=1.0,
        region_annotation_loss_type="pred_region",
        region_annotation_pred_kernel_size=5,
        region_annotation_focal_loss_lambda=0,
        region_annotation_contrast_alpha=0.3,
        regularization_lambda=0.5,
        use_adv=False,
        use_mmd_adv=False,
        add_repulsive_mmd=False,
        use_temporal_mmd=False,
        temporal_mmd_cache_size=32,
        temporal_mmd_discount_factor=0.60,
        adv_loss_lambda=0.5,
        train_adv_seperate=False,
        anneal_adv_loss=False,
        turn_off_model_train=False,
        adv_on_logits_alone=False,
        num_model_steps=1,
        num_adv_steps=100,
        wrap_model=False,
        use_risk_factors=False,
        pred_risk_factors=True,
        pred_risk_factors_lambda=0.25,
        use_pred_risk_factors_at_test=True,
        use_pred_risk_factors_if_unk=False,
        risk_factor_keys=[],
        risk_factor_metadata_path="",
        survival_analysis_setup=True,
        make_probs_indep=False,
        mask_mechanism="default",
        eval_survival_on_risk=False,
        max_followup=5,
        eval_risk_survival=False,
        mask_prob=0,
        pred_missing_mammos=False,
        also_pred_given_mammos=False,
        pred_missing_mammos_lambda=0.25,
        use_precomputed_hiddens=False,
        zero_out_hiddens=False,
        use_precomputed_hiddens_in_get_hiddens=False,
        hiddens_results_path="",
        use_dev_to_train_model_on_hiddens=False,
        turn_off_init_projection=False,
        optimizer="adam",
        objective="cross_entropy",
        init_lr=0.001,
        momentum=0,
        lr_decay=0.5,
        weight_decay=0,
        patience=10,
        turn_off_model_reset=False,
        tuning_metric="loss",
        epochs=256,
        max_batches_per_train_epoch=10000,
        max_batches_per_dev_epoch=10000,
        batch_size=batch_size,
        batch_splits=1,
        dropout=0.25,
        save_dir="snapshot",
        results_path="logs/snapshot",
        prediction_save_path=None,
        no_tuning_on_dev=False,
        lr_reduction_interval=1,
        data_fraction=1.0,
        ten_fold_cross_val=False,
        ten_fold_cross_val_seed=1,
        ten_fold_test_index=0,
        model_name="mirai_full",
        num_layers=3,
        snapshot=None,
        state_dict_path=None,
        img_encoder_snapshot=encoder_path,
        freeze_image_encoder=False,
        transformer_snapshot=transformer_path,
        callibrator_snapshot=calibrator_path,
        patch_snapshot=None,
        pretrained_on_imagenet=False,
        pretrained_imagenet_model_name="resnet18",
        make_fc=False,
        replace_bn_with_gn=False,
        block_layout=[
            "BasicBlock,2",
            "BasicBlock,2",
            "BasicBlock,2",
            "BasicBlock,2",
        ],
        block_widening_factor=1,
        num_groups=1,
        pool_name="GlobalAvgPool",
        deep_risk_factor_pool=False,
        replace_snapshot_pool=False,
        is_ccds_server=False,
        cuda=use_cuda and torch.cuda.is_available(),
        num_gpus=1,
        num_shards=1,
        data_parallel=False,
        model_parallel=False,
        plot_losses=False,
        cluster_exams=False,
        background_size=[1024, 1024],
        noise=False,
        noise_var=0.1,
        use_permissive_cohort=True,
        mammogram_type=None,
        resume=False,
        ignore_warnings=True,
        optimizer_state=None,
        current_epoch=None,
        lr=None,
        epoch_stats=None,
        step_indx=1,
        censoring_distribution=None,
    )

    dataset_factory.get_dataset_class(args).set_args(args)
    args.device = "cuda" if args.cuda else "cpu"
    args.image_transformers = _parse_transformers(args.image_transformers)
    args.tensor_transformers = _parse_transformers(args.tensor_transformers)
    args.test_image_transformers = _parse_transformers(args.test_image_transformers)
    args.test_tensor_transformers = _parse_transformers(args.test_tensor_transformers)
    args.block_layout = _parse_block_layout(args.block_layout)
    _validate_args(args)
    return args


def _rows_to_df(exams, probs, max_followup, calibrator):
    rows = []
    for exam, arr in zip(exams, probs):
        row = {"patient_exam_id": exam}
        for index in range(max_followup):
            key = "{}_year_risk".format(index + 1)
            raw_val = arr[index]
            if calibrator is not None:
                row[key] = calibrator[index].predict_proba([[raw_val]])[0, 1]
            else:
                row[key] = raw_val
        rows.append(row)
    columns = ["patient_exam_id"] + [
        "{}_year_risk".format(index + 1) for index in range(max_followup)
    ]
    return pd.DataFrame(rows, columns=columns)


def load_model(
    model_dir,
    calibrate=True,
    batch_size=1,
    img_mean=7047.99,
    img_std=12005.5,
    img_size=(1664, 2048),
    num_workers=0,
    use_cuda=False,
):
    args = _build_args(
        metadata_csv=os.devnull,
        model_dir=model_dir,
        calibrate=calibrate,
        batch_size=batch_size,
        img_mean=img_mean,
        img_std=img_std,
        img_size=img_size,
        num_workers=num_workers,
        use_cuda=use_cuda,
    )
    model = model_factory.get_model(args)
    calibrator = _load_calibrator(args.callibrator_snapshot) if calibrate else None
    print(model)
    return {
        "args": args,
        "model": model,
        "calibrator": calibrator,
    }


def predict_loaded(loaded, metadata_csv, output_csv=None):
    args = copy.deepcopy(loaded["args"])
    args.metadata_path = metadata_csv
    if output_csv is not None:
        args.prediction_save_path = output_csv

    transformers = transformer_factory.get_transformers(
        args.image_transformers, args.tensor_transformers, args
    )
    test_transformers = transformer_factory.get_transformers(
        args.test_image_transformers, args.test_tensor_transformers, args
    )
    _, _, test_data = dataset_factory.get_dataset(args, transformers, test_transformers)
    test_stats = train.eval_model(test_data, loaded["model"], args)
    exams = test_stats["exams"]
    probs = test_stats["probs"]
    df = _rows_to_df(
        exams=exams,
        probs=probs,
        max_followup=args.max_followup,
        calibrator=loaded.get("calibrator"),
    )
    if output_csv:
        df.to_csv(output_csv, index=False)
        print("Saved predictions to {}".format(output_csv))
    return df


def predict(
    metadata_csv,
    model_dir,
    calibrate=True,
    output_csv=None,
    batch_size=1,
    img_mean=7047.99,
    img_std=12005.5,
    img_size=(1664, 2048),
    num_workers=0,
    use_cuda=False,
):
    loaded = load_model(
        model_dir=model_dir,
        calibrate=calibrate,
        batch_size=batch_size,
        img_mean=img_mean,
        img_std=img_std,
        img_size=img_size,
        num_workers=num_workers,
        use_cuda=use_cuda,
    )
    return predict_loaded(loaded, metadata_csv=metadata_csv, output_csv=output_csv)
