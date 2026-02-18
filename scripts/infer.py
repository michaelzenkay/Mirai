"""Clean programmatic inference API for Mirai breast cancer risk prediction.

Usage:
    from mirai.scripts.infer import predict

    df = predict(
        metadata_csv="path/to/metadata.csv",
        model_dir="path/to/models/",
    )
    # df has columns: patient_exam_id, 1_year_risk, ..., 5_year_risk
"""
import pickle
import os
import sys
from os.path import dirname, realpath, join, abspath
from glob import glob
from argparse import Namespace

# Ensure mirai root is on path
_MIRAI_ROOT = dirname(dirname(realpath(__file__)))
if _MIRAI_ROOT not in sys.path:
    sys.path.insert(0, _MIRAI_ROOT)

import torch
import pandas as pd
import onconet.datasets.factory as dataset_factory
import onconet.models.factory as model_factory
import onconet.transformers.factory as transformer_factory
from onconet.learn import train
from onconet.datasets.factory import DATASET_REGISTRY
from onconet.datasets.csv_mammo_cancer import CSV_Mammo_Cancer_Survival_All_Images_Dataset
from onconet.utils.parsing import parse_transformers, parse_block_layout, validate_args

# Register the dataset class needed for inference
DATASET_REGISTRY['csv_mammo_risk_all_full_future'] = CSV_Mammo_Cancer_Survival_All_Images_Dataset


def _find_file(directory, pattern):
    """Find a single file matching a glob pattern in directory."""
    matches = glob(join(directory, pattern))
    if not matches:
        return None
    return matches[0]


def _build_args(metadata_csv, model_dir, calibrate=True, batch_size=1,
                img_mean=7047.99, img_std=12005.5, img_size=(1664, 2048),
                num_workers=0, use_cuda=False):
    """Build a Namespace matching what parsing.parse_args() would produce."""

    # Auto-detect model files
    encoder_path = _find_file(model_dir, "*Base*")
    transformer_path = _find_file(model_dir, "*Transformer*")
    calibrator_path = None
    if calibrate:
        # Prefer the newer calibrator that works with modern sklearn
        calibrator_path = _find_file(model_dir, "*mar12_2022*callibrator*") or \
                          _find_file(model_dir, "*callibrator_mar*") or \
                          _find_file(model_dir, "*callibrator*")

    if encoder_path is None:
        raise FileNotFoundError("No image encoder snapshot (*Base*) found in {}".format(model_dir))
    if transformer_path is None:
        raise FileNotFoundError("No transformer snapshot (*Transformer*) found in {}".format(model_dir))

    args = Namespace(
        # Run mode
        train=False,
        test=True,
        dev=False,
        resume=False,
        ignore_warnings=True,
        get_dataset_stats=False,

        # Data
        dataset='csv_mammo_risk_all_full_future',
        metadata_path=metadata_csv,
        metadata_dir=None,
        img_dir='',
        img_size=list(img_size),
        img_mean=[img_mean],
        img_std=[img_std],
        num_chan=3,
        num_workers=num_workers,
        cache_path=None,
        patch_size=[-1, -1],
        batch_size=batch_size,
        batch_splits=1,

        # Model
        model_name='mirai_full',
        snapshot=None,
        state_dict_path=None,
        img_encoder_snapshot=encoder_path,
        transformer_snapshot=transformer_path,
        callibrator_snapshot=calibrator_path,
        patch_snapshot=None,
        replace_snapshot_pool=False,
        pretrained_on_imagenet=False,
        pretrained_imagenet_model_name='resnet18',
        make_fc=False,
        replace_bn_with_gn=False,
        wrap_model=False,
        freeze_image_encoder=False,

        # Model architecture
        num_layers=3,
        input_dim=512,
        transfomer_hidden_dim=512,
        num_heads=8,
        block_layout=["BasicBlock,2", "BasicBlock,2", "BasicBlock,2", "BasicBlock,2"],
        block_widening_factor=1,
        num_groups=1,
        pool_name='GlobalAvgPool',
        deep_risk_factor_pool=False,
        dropout=0.25,

        # Multi-image
        multi_image=True,
        num_images=4,
        min_num_images=4,
        video=False,
        pred_both_sides=False,

        # Device
        cuda=use_cuda,
        num_gpus=1,
        num_shards=1,
        data_parallel=False,
        model_parallel=False,

        # Survival analysis
        survival_analysis_setup=True,
        max_followup=5,
        eval_survival_on_risk=False,
        eval_risk_survival=False,
        make_probs_indep=False,
        mask_mechanism='default',

        # Risk factors
        use_risk_factors=False,
        pred_risk_factors=True,
        pred_risk_factors_lambda=0.25,
        use_pred_risk_factors_at_test=True,
        use_pred_risk_factors_if_unk=False,
        risk_factor_keys=['density', 'binary_family_history', 'binary_biopsy_benign',
                          'binary_biopsy_LCIS', 'binary_biopsy_atypical_hyperplasia',
                          'age', 'menarche_age', 'menopause_age', 'first_pregnancy_age',
                          'prior_hist', 'race', 'parous', 'menopausal_status',
                          'weight', 'height', 'ovarian_cancer', 'ovarian_cancer_age',
                          'ashkenazi', 'brca', 'mom_bc_cancer_history',
                          'm_aunt_bc_cancer_history', 'p_aunt_bc_cancer_history',
                          'm_grandmother_bc_cancer_history', 'p_grantmother_bc_cancer_history',
                          'sister_bc_cancer_history', 'mom_oc_cancer_history',
                          'm_aunt_oc_cancer_history', 'p_aunt_oc_cancer_history',
                          'm_grandmother_oc_cancer_history', 'p_grantmother_oc_cancer_history',
                          'sister_oc_cancer_history', 'hrt_type', 'hrt_duration',
                          'hrt_years_ago_stopped'],
        risk_factor_metadata_path='',

        # Eval
        cluster_exams=False,
        confidence_interval=0.95,
        num_resamples=10000,
        threshold=None,
        predict_birads=False,
        predict_birads_lambda=0,

        # Sampling
        class_bal=True,
        year_weighted_class_bal=False,
        shift_class_bal_towards_imediate_cancers=False,
        device_class_bal=False,
        allowed_devices='all',
        use_c_view_if_available=False,

        # Losses / regularization
        objective='cross_entropy',
        use_region_annotation=False,
        fraction_region_annotation_to_use=1.0,
        region_annotation_loss_type='pred_region',
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

        # Optimizer (unused at inference but needed for validation)
        optimizer='adam',
        init_lr=0.001,
        momentum=0,
        lr_decay=0.5,
        weight_decay=0,
        patience=10,
        turn_off_model_reset=False,
        tuning_metric='loss',
        no_tuning_on_dev=False,
        lr_reduction_interval=1,

        # Training params (unused but needed)
        epochs=256,
        max_batches_per_train_epoch=10000,
        max_batches_per_dev_epoch=10000,
        data_fraction=1.0,
        train_years=[],
        dev_years=[],
        test_years=[],
        run_prefix='snapshot',
        save_dir='snapshot',
        results_path='logs/snapshot',
        prediction_save_path=None,

        # Ten fold
        ten_fold_cross_val=False,
        ten_fold_cross_val_seed=1,
        ten_fold_test_index=0,

        # Visualization
        plot_losses=False,

        # Dataset specific
        background_size=[1024, 1024],
        noise=False,
        noise_var=0.1,
        use_permissive_cohort=True,
        mammogram_type=None,
        invasive_only=False,
        rebalance_eval_cancers=False,
        downsample_activ=False,
        ensemble_paths=[],
        drop_benign_side=False,

        # Spatial transformer
        use_spatial_transformer=False,
        spatial_transformer_name='affine',
        spatial_transformer_img_size=[208, 256],
        location_network_name='resnet18',
        location_network_block_layout=["BasicBlock,2", "BasicBlock,2",
                                        "BasicBlock,2", "BasicBlock,2"],
        tps_grid_size=10,
        tps_span_range=0.9,

        # Hiddens
        use_precomputed_hiddens=False,
        zero_out_hiddens=False,
        use_precomputed_hiddens_in_get_hiddens=False,
        hiddens_results_path='',
        use_dev_to_train_model_on_hiddens=False,
        turn_off_init_projection=False,
        force_input_dim=False,
        get_activs_instead_of_hiddens=False,
        is_ccds_server=False,

        # Generative
        mask_prob=0,
        pred_missing_mammos=False,
        also_pred_given_mammos=False,
        pred_missing_mammos_lambda=0.25,

        # Censoring distribution (computed from training data, not needed for inference)
        censoring_distribution=None,

        # State (set by parse_args)
        optimizer_state=None,
        current_epoch=None,
        lr=None,
        epoch_stats=None,
        step_indx=1,
    )

    # Set device
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'

    # Let the dataset class override args (sets survival_analysis_setup, transformers, etc.)
    dataset_factory.get_dataset_class(args).set_args(args)

    # Parse list args (transformers, block_layout)
    args.image_transformers = parse_transformers(args.image_transformers)
    args.tensor_transformers = parse_transformers(args.tensor_transformers)
    args.test_image_transformers = parse_transformers(args.test_image_transformers)
    args.test_tensor_transformers = parse_transformers(args.test_tensor_transformers)
    args.block_layout = parse_block_layout(args.block_layout)

    validate_args(args)

    return args


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


def _load_calibrator(path):
    """Load calibrator with graceful error handling for sklearn version mismatches."""
    if path is None:
        return None
    try:
        calibrator = pickle.load(open(path, 'rb'))
        _patch_calibrator(calibrator)
        return calibrator
    except ModuleNotFoundError as e:
        print("WARNING: Could not load calibrator from {}".format(path))
        print("  Cause: {}".format(e))
        print("  Try using Mirai_pred_rf_callibrator_mar12_2022.p instead.")
        return None


def predict(metadata_csv, model_dir, calibrate=True, output_csv=None,
            batch_size=1, img_mean=7047.99, img_std=12005.5,
            img_size=(1664, 2048), num_workers=0, use_cuda=False):
    """Run Mirai inference on a metadata CSV.

    Args:
        metadata_csv: Path to CSV with columns:
            patient_id, exam_id, laterality, view, file_path,
            years_to_cancer, years_to_last_followup, split_group
        model_dir: Directory containing model files. Auto-detects:
            - *Base* (image encoder)
            - *Transformer* (temporal transformer)
            - *callibrator* (calibrator, prefers mar12_2022 version)
        calibrate: Apply Platt scaling calibration (default True).
        output_csv: Optional path to save results CSV.
        batch_size: Inference batch size (default 1).
        img_mean: Image pixel mean for normalization.
        img_std: Image pixel std for normalization.
        img_size: Tuple of (height, width) for image scaling.
        num_workers: DataLoader workers (default 0 for Windows compat).
        use_cuda: Whether to use GPU.

    Returns:
        pd.DataFrame with columns:
            patient_exam_id, 1_year_risk, 2_year_risk, ..., 5_year_risk
    """
    args = _build_args(
        metadata_csv=metadata_csv,
        model_dir=model_dir,
        calibrate=calibrate,
        batch_size=batch_size,
        img_mean=img_mean,
        img_std=img_std,
        img_size=img_size,
        num_workers=num_workers,
        use_cuda=use_cuda,
    )

    # Load transformers
    transformers = transformer_factory.get_transformers(
        args.image_transformers, args.tensor_transformers, args)
    test_transformers = transformer_factory.get_transformers(
        args.test_image_transformers, args.test_tensor_transformers, args)

    # Load dataset
    _, _, test_data = dataset_factory.get_dataset(args, transformers, test_transformers)

    # Load model
    model = model_factory.get_model(args)
    print(model)

    # Run inference — returns stats dict with 'exams' and 'probs'
    test_stats = train.eval_model(test_data, model, args)
    exams = test_stats['exams']
    probs = test_stats['probs']

    # Load calibrator
    calibrator = None
    if calibrate and args.callibrator_snapshot:
        calibrator = _load_calibrator(args.callibrator_snapshot)

    # Build results
    rows = []
    for exam, arr in zip(exams, probs):
        row = {'patient_exam_id': exam}
        for i in range(args.max_followup):
            key = "{}_year_risk".format(i + 1)
            raw_val = arr[i]
            if calibrator is not None:
                row[key] = calibrator[i].predict_proba([[raw_val]])[0, 1]
            else:
                row[key] = raw_val
        rows.append(row)

    columns = ['patient_exam_id'] + ["{}_year_risk".format(i + 1) for i in range(args.max_followup)]
    df = pd.DataFrame(rows, columns=columns)

    if output_csv:
        df.to_csv(output_csv, index=False)
        print("Saved predictions to {}".format(output_csv))

    return df
