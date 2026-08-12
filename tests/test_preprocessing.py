import numpy as np

from src.data import load_config, make_synthetic_fraud_data, prepare_dataset


def test_temporal_split_and_shapes():
    config = load_config("configs/ieee_cis.yaml")
    frame = make_synthetic_fraud_data(1000, seed=7)
    prepared = prepare_dataset(frame, config)
    assert prepared.train_frame["TransactionDT"].max() <= prepared.validation_frame["TransactionDT"].min()
    assert prepared.validation_frame["TransactionDT"].max() <= prepared.test_frame["TransactionDT"].min()
    assert prepared.X_train.shape[0] == len(prepared.y_train)
    assert prepared.X_train.shape[1] == prepared.X_validation.shape[1] == prepared.X_test.shape[1]
    assert np.isfinite(prepared.X_train).all()


def test_preprocessor_is_fitted_on_train_only():
    config = load_config("configs/ieee_cis.yaml")
    frame = make_synthetic_fraud_data(1000, seed=9)
    prepared = prepare_dataset(frame, config)
    numeric_pipeline = prepared.preprocessor.transformer.named_transformers_["numeric"]
    scaler = numeric_pipeline.named_steps["scaler"]
    numeric = prepared.preprocessor.numeric_features
    imputer = numeric_pipeline.named_steps["imputer"]
    train_imputed = imputer.transform(prepared.train_frame[numeric])
    assert np.allclose(scaler.mean_, train_imputed.mean(axis=0))
