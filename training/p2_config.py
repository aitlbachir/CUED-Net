"""Single source for every Protocol P2 setting reported in the paper."""

SEEDS = [42, 123, 456, 789, 2024]
OUTER_FOLDS = [0, 1, 2, 3, 4]
TEST_PHASE_FOLD = -1

TRAINING = dict(
    max_epochs=50,
    patience=15,
    freeze_epochs=5,
    batch_size=16,
    lr_head=1e-4,
    lr_encoder=1e-5,
    weight_decay=1e-4,
    cosine_t0=10,
    cosine_tmult=2,
    cosine_eta_min=1e-6,
    grad_clip=1.0,
    selection_f1_weight=0.6,
    selection_auc_weight=0.4,
)

LOSS = dict(lambda_kl=0.1, lambda_consistency=0.1, annealing_epochs=10)
MC_DROPOUT_PASSES = 50
DECISION_THRESHOLD = 0.5

# CMMD head adaptation. Fractions are of patients, stratified, seed 2026.
ADAPTATION = dict(train_fraction=0.10, val_fraction=0.20, max_epochs=30, patience=8, lr=1e-4, weight_decay=1e-4, batch_size=16)
ADAPTED_METHODS = ["cued_net", "tmc"]
LOCKED_METHODS = ["cued_net", "softmax_dual", "tmc", "sv_edl", "cross_view", "cued_convnext"]

# method id -> constructor settings. Each table in the paper names these ids.
METHODS = {
    "cued_net": dict(model="cued_net", backbone="densenet121", kl_mode="standard", save_ckpt=True),
    "softmax_dual": dict(model="softmax_dual", save_ckpt=True, mc_dropout=True),
    "tmc": dict(model="tmc", save_ckpt=True),
    "sv_edl": dict(model="sv_edl", save_ckpt=True),
    "cross_view": dict(model="cross_view", save_ckpt=True),
    "cued_r1_kl": dict(model="cued_net", backbone="densenet121", kl_mode="r1", save_ckpt=False),
    "cued_convnext": dict(model="cued_net", backbone="convnext_tiny", kl_mode="standard", save_ckpt=True),
    "cued_resnet50": dict(model="cued_net", backbone="resnet50", kl_mode="standard", save_ckpt=False),
    "cued_effnet_b0": dict(model="cued_net", backbone="efficientnet_b0", kl_mode="standard", save_ckpt=False),
    "cued_no_consistency": dict(model="cued_net", backbone="densenet121", kl_mode="standard", lambda_consistency=0.0, save_ckpt=False),
    "cued_vdl": dict(model="cued_net", backbone="densenet121", kl_mode="standard", lambda_vdl=0.3, save_ckpt=False),
}
