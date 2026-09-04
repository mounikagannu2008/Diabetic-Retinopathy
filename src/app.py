"""
Local button-driven UI for the DR screening Level-1 pipeline.

Run with:
    streamlit run src/app.py

This wraps make_splits.py / train.py / finetune.py / evaluate.py / inference.py
as buttons — it does not reimplement their logic, so behavior stays
identical to the CLI scripts and there's a single source of truth.
"""

import os
import sys
import json
import subprocess
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from config import cfg

st.set_page_config(page_title="DR Screening — Level 1", layout="wide")
st.title("Diabetic Retinopathy Screening — Level 1")
st.caption("EfficientNet-B3 transfer learning · CLAHE + illumination-normalized preprocessing · Grad-CAM explainability")

tab_setup, tab_train, tab_finetune, tab_evaluate, tab_predict = st.tabs(
    ["1. Data Setup", "2. Train", "3. Fine-tune", "4. Evaluate", "5. Predict"]
)

SRC_DIR = os.path.dirname(__file__)


def run_script(script_name: str, args: list, log_placeholder):
    """Run a pipeline script as a subprocess and stream its output live."""
    cmd = [sys.executable, os.path.join(SRC_DIR, script_name)] + args
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=os.path.dirname(SRC_DIR),
    )
    log_lines = []
    for line in process.stdout:
        log_lines.append(line.rstrip())
        log_placeholder.code("\n".join(log_lines[-40:]))  # show last 40 lines, live
    process.wait()
    return process.returncode, "\n".join(log_lines)


# ---------------------------------------------------------------------
# TAB 1: DATA SETUP
# ---------------------------------------------------------------------
with tab_setup:
    st.header("Point the pipeline at your dataset")
    st.write(
        "Edit `src/config.py` (labels_csv, images_dir, image_ext) to match where "
        "you downloaded the dataset, or just confirm the current settings below."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Labels CSV", value=cfg.labels_csv, disabled=True)
        st.text_input("Images directory", value=cfg.images_dir, disabled=True)
    with col2:
        st.text_input("Image extension", value=cfg.image_ext, disabled=True)
        st.text_input("Splits output directory", value=cfg.splits_dir, disabled=True)

    csv_exists = os.path.exists(cfg.labels_csv)
    images_exist = os.path.exists(cfg.images_dir)
    st.write(f"Labels CSV found: {'✅' if csv_exists else '❌ — edit config.py'}")
    st.write(f"Images directory found: {'✅' if images_exist else '❌ — edit config.py'}")

    if st.button("Create stratified train / val / test split", type="primary"):
        log_placeholder = st.empty()
        code, _ = run_script("make_splits.py", [], log_placeholder)
        if code == 0:
            st.success("Splits created in data/splits/")
        else:
            st.error("Split creation failed — see log above.")

    splits_dir = cfg.splits_dir
    if all(os.path.exists(os.path.join(splits_dir, f"{s}.csv")) for s in ["train", "val", "test"]):
        st.subheader("Current splits")
        for split_name in ["train", "val", "test"]:
            df = pd.read_csv(os.path.join(splits_dir, f"{split_name}.csv"))
            st.write(f"**{split_name}**: {len(df)} images — class distribution:")
            st.bar_chart(df["diagnosis"].value_counts().sort_index())


# ---------------------------------------------------------------------
# TAB 2: TRAIN
# ---------------------------------------------------------------------
with tab_train:
    st.header("Train EfficientNet-B3 from ImageNet weights")
    st.write(f"Backbone: `{cfg.backbone}` · Epochs: {cfg.num_epochs} (with early stopping, "
             f"patience={cfg.early_stopping_patience}) · Batch size: {cfg.batch_size}")
    st.write("Adjust hyperparameters in `src/config.py` before running if needed.")

    if st.button("Start Training", type="primary"):
        splits_ready = all(
            os.path.exists(os.path.join(cfg.splits_dir, f"{s}.csv")) for s in ["train", "val", "test"]
        )
        if not splits_ready:
            st.error("No splits found — go to the 'Data Setup' tab and create splits first.")
        else:
            log_placeholder = st.empty()
            with st.spinner("Training in progress — this can take a while depending on dataset size and hardware..."):
                code, full_log = run_script("train.py", [], log_placeholder)
            if code == 0:
                st.success("Training complete. Best checkpoint saved to models/dr_classifier_best.pt")
            else:
                st.error("Training failed — see log above.")

    history_path = os.path.join(cfg.output_dir, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        hist_df = pd.DataFrame(history).set_index("epoch")
        st.subheader("Training history (last run)")
        st.line_chart(hist_df[["train_f1", "val_f1"]])
        st.line_chart(hist_df[["train_loss", "val_loss"]])


# ---------------------------------------------------------------------
# TAB 3: FINE-TUNE
# ---------------------------------------------------------------------
with tab_finetune:
    st.header("Continue training an existing checkpoint")
    st.write("Use this after an initial training run has plateaued, or after adding more data.")

    model_files = [f for f in os.listdir(cfg.model_dir) if f.endswith(".pt")] if os.path.exists(cfg.model_dir) else []
    if not model_files:
        st.info("No checkpoints found yet — train a model first.")
    else:
        checkpoint_choice = st.selectbox("Checkpoint to fine-tune from", model_files)
        col1, col2, col3 = st.columns(3)
        with col1:
            ft_epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10)
        with col2:
            ft_lr_backbone = st.number_input("Backbone LR", min_value=1e-7, max_value=1e-3,
                                              value=5e-6, format="%.7f")
        with col3:
            ft_lr_head = st.number_input("Head LR", min_value=1e-6, max_value=1e-2,
                                          value=5e-5, format="%.6f")

        if st.button("Start Fine-tuning", type="primary"):
            log_placeholder = st.empty()
            checkpoint_path = os.path.join(cfg.model_dir, checkpoint_choice)
            args = [
                "--checkpoint", checkpoint_path,
                "--epochs", str(ft_epochs),
                "--lr_backbone", str(ft_lr_backbone),
                "--lr_head", str(ft_lr_head),
            ]
            with st.spinner("Fine-tuning in progress..."):
                code, _ = run_script("finetune.py", args, log_placeholder)
            if code == 0:
                st.success("Fine-tuning run complete — see log for whether it improved on the starting checkpoint.")
            else:
                st.error("Fine-tuning failed — see log above.")


# ---------------------------------------------------------------------
# TAB 4: EVALUATE
# ---------------------------------------------------------------------
with tab_evaluate:
    st.header("Evaluate on the held-out test set")

    model_files = [f for f in os.listdir(cfg.model_dir) if f.endswith(".pt")] if os.path.exists(cfg.model_dir) else []
    if not model_files:
        st.info("No checkpoints found yet — train a model first.")
    else:
        eval_checkpoint = st.selectbox("Checkpoint to evaluate", model_files, key="eval_checkpoint")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Run Evaluation", type="primary"):
                log_placeholder = st.empty()
                checkpoint_path = os.path.join(cfg.model_dir, eval_checkpoint)
                with st.spinner("Running inference over the test set..."):
                    code, _ = run_script("evaluate.py", ["--checkpoint", checkpoint_path], log_placeholder)
                if code != 0:
                    st.error("Evaluation failed — see log above.")
        with col_b:
            if st.button("Tune Referable-DR Threshold"):
                st.caption("Calibrates the referable/non-referable decision on the validation "
                           "set to better balance sensitivity and specificity — doesn't retrain "
                           "the model, and doesn't touch the test set until reporting the result.")
                log_placeholder = st.empty()
                checkpoint_path = os.path.join(cfg.model_dir, eval_checkpoint)
                with st.spinner("Scanning thresholds on validation set, then reporting on test set..."):
                    code, _ = run_script("tune_threshold.py", ["--checkpoint", checkpoint_path], log_placeholder)
                if code == 0:
                    st.success("Threshold tuned and saved. Future Evaluate/Predict runs will use it automatically.")
                else:
                    st.error("Threshold tuning failed — see log above.")

        tuned_threshold_path = os.path.join(cfg.output_dir, "tuned_threshold.json")
        if os.path.exists(tuned_threshold_path):
            with open(tuned_threshold_path) as f:
                tuned_info = json.load(f)
            st.info(f"Active tuned threshold: **{tuned_info['threshold']:.2f}** "
                    f"(from validation set — sensitivity={tuned_info['val_sensitivity_at_tuning']:.3f}, "
                    f"specificity={tuned_info['val_specificity_at_tuning']:.3f} at tuning time)")
        else:
            st.caption("No tuned threshold yet — Evaluate/Predict currently use the model's plain "
                       "argmax decision for referable status.")

    report_path = os.path.join(cfg.output_dir, "evaluation_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)

        st.subheader("Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{report['overall_accuracy']:.3f}")
        m2.metric("Macro F1", f"{report['macro_f1']:.3f}")
        ref = report["referable_dr_metrics"]
        m3.metric("Referable Sensitivity", f"{ref['referable_sensitivity']:.3f}",
                   delta="Target: >0.90", delta_color="off")
        m4.metric("Referable Specificity", f"{ref['referable_specificity']:.3f}",
                   delta="Target: >0.85", delta_color="off")

        st.write(f"Referable DR ROC-AUC: **{ref['referable_roc_auc']:.3f}**")
        st.write(f"Sensitivity target met: {'✅' if ref['meets_sensitivity_target_90pct'] else '❌'} · "
                 f"Specificity target met: {'✅' if ref['meets_specificity_target_85pct'] else '❌'}")
        st.caption(f"Decision rule: {ref.get('decision_rule_used', 'argmax')}")

        st.subheader("Per-class metrics")
        st.dataframe(pd.DataFrame(report["per_class_metrics"]).T)

        cm_path = os.path.join(cfg.output_dir, "confusion_matrix.png")
        roc_path = os.path.join(cfg.output_dir, "roc_curves.png")
        col1, col2 = st.columns(2)
        if os.path.exists(cm_path):
            col1.image(cm_path, caption="Confusion matrix")
        if os.path.exists(roc_path):
            col2.image(roc_path, caption="ROC curve — referable DR")

        st.caption(report["note"])


# ---------------------------------------------------------------------
# TAB 5: PREDICT (single image + Grad-CAM)
# ---------------------------------------------------------------------
with tab_predict:
    st.header("Single-image inference with explanation")

    model_files = [f for f in os.listdir(cfg.model_dir) if f.endswith(".pt")] if os.path.exists(cfg.model_dir) else []
    if not model_files:
        st.info("No checkpoints found yet — train a model first.")
    else:
        predict_checkpoint = st.selectbox("Checkpoint", model_files, key="predict_checkpoint")
        uploaded_image = st.file_uploader("Upload a fundus image", type=["png", "jpg", "jpeg"])

        if uploaded_image is not None:
            temp_path = os.path.join(cfg.output_dir, "_temp_predict_input.png")
            with open(temp_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

            st.image(temp_path, caption="Uploaded image", width=300)

            if st.button("Run Prediction", type="primary"):
                checkpoint_path = os.path.join(cfg.model_dir, predict_checkpoint)
                log_placeholder = st.empty()
                with st.spinner("Running inference and generating Grad-CAM explanation..."):
                    code, log = run_script(
                        "inference.py",
                        ["--image", temp_path, "--checkpoint", checkpoint_path],
                        log_placeholder,
                    )

                if code == 0:
                    try:
                        json_start = log.index("{")
                        json_end = log.rindex("}") + 1
                        result = json.loads(log[json_start:json_end])

                        st.subheader("Prediction")
                        c1, c2 = st.columns(2)
                        c1.metric("DR Level", f"{result['predicted_level']} — {result['predicted_class_name']}")
                        c2.metric("Confidence", f"{result['confidence']:.1%}")

                        st.write(f"**Referable status:** {result['referable_label']} "
                                 f"(probability: {result['referable_probability']:.1%})")
                        st.caption(f"Decision rule: {result.get('referable_decision_rule', 'argmax')}")

                        st.write("**Class probabilities:**")
                        probs_df = pd.DataFrame(
                            list(result["class_probabilities"].items()), columns=["Class", "Probability"]
                        )
                        st.bar_chart(probs_df.set_index("Class"))

                        if not result["quality_check"]["likely_gradeable"]:
                            st.warning("Image failed basic quality heuristics (blur/brightness) — "
                                       "prediction may be unreliable.")

                        base_name = os.path.splitext(os.path.basename(temp_path))[0]
                        overlay_path = os.path.join(cfg.output_dir, "example_predictions",
                                                     f"{base_name}_gradcam_overlay.png")
                        preproc_path = os.path.join(cfg.output_dir, "example_predictions",
                                                     f"{base_name}_preprocessed.png")
                        if os.path.exists(overlay_path):
                            col1, col2 = st.columns(2)
                            col1.image(preproc_path, caption="Preprocessed input")
                            col2.image(overlay_path, caption="Grad-CAM explanation")
                    except (ValueError, json.JSONDecodeError):
                        st.warning("Could not parse structured result — see raw log below.")
                else:
                    st.error("Inference failed — see log above.")
