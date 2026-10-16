"""NeuroScan - Brain Tumor MRI classification dashboard.  Run: streamlit run app.py"""
import json
import random
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots

import config
import data_utils

st.set_page_config(page_title="NeuroScan - Brain MRI Classifier", page_icon="🧠", layout="wide")

# Palette: light-box grey, ink, radiology teal, review amber
INK, TEAL, AMBER, MIST = "#14202B", "#0E7C86", "#C2410C", "#DCE4E8"
COLORWAY = [TEAL, "#334E68", AMBER, "#7C9CB2", "#8A6F2A"]

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
.masthead {{ border-left: 6px solid {TEAL}; padding: .2rem 0 .2rem 1rem; margin-bottom: 1rem; }}
.masthead h1 {{ margin: 0; font-size: 2rem; font-weight: 600; color: {INK}; }}
.masthead p {{ margin: .2rem 0 0; color: #52606D; }}
.panel {{ background: #fff; border: 1px solid {MIST}; border-radius: 8px; padding: 1rem 1.2rem; }}
.verdict {{ border-radius: 8px; padding: 1rem 1.2rem; color: #fff; font-size: 1.05rem; }}
.verdict b {{ font-size: 1.5rem; display: block; }}
.tag {{ display:inline-block; padding:.1rem .6rem; border-radius:4px; font-size:.8rem; color:#fff; }}
[data-testid="stMetricValue"] {{ font-weight: 600; }}
</style>
""",
    unsafe_allow_html=True,
)

PRETTY = {"custom_cnn": "Custom CNN", "resnet50": "ResNet50", "mobilenetv2": "MobileNetV2",
          "inceptionv3": "InceptionV3", "efficientnetb0": "EfficientNetB0"}
CLASS_INFO = {
    "glioma": ("Glioma", "Arises from glial cells; often infiltrates surrounding brain tissue.", "High", "#B42318"),
    "meningioma": ("Meningioma", "Grows from the meninges; usually slow-growing and often benign.", "Medium", "#B54708"),
    "pituitary": ("Pituitary tumor", "Sits in the pituitary gland; can affect hormones and vision.", "Medium", "#6941C6"),
    "notumor": ("No tumor", "No tumor pattern detected in this scan.", "Low", "#067647"),
}


def pretty(name):
    return PRETTY.get(name, name)


def class_key(name):
    return re.sub(r"[^a-z]", "", name.lower())


def class_info(name):
    return CLASS_INFO.get(class_key(name), (name.title(), "Tumor category from the dataset.", "Review", TEAL))


# ------------------------------------------------------------------ cached loaders
@st.cache_data(ttl=60)
def load_class_names():
    f = config.REPORTS_DIR / "class_names.json"
    if f.exists():
        return json.loads(f.read_text())
    train = data_utils.get_split_dirs()["train"]
    return list(data_utils.class_counts(train)) if train else []


@st.cache_data(ttl=60)
def load_metrics():
    return {p.name.replace("_metrics.json", ""): json.loads(p.read_text())
            for p in sorted(config.REPORTS_DIR.glob("*_metrics.json"))}


@st.cache_data(ttl=60)
def load_history(name):
    f = config.REPORTS_DIR / f"{name}_history.json"
    return json.loads(f.read_text()) if f.exists() else None


@st.cache_data(ttl=60)
def split_counts():
    rows = []
    for split, path in data_utils.get_split_dirs().items():
        if path:
            rows += [{"Split": split, "Class": c, "Images": n} for c, n in data_utils.class_counts(path).items()]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Sampling images...")
def image_stats(files):
    rows = []
    for f in files:
        with Image.open(f) as im:
            g = np.asarray(im.convert("L"), dtype="float32")
            rows.append({"Width": im.width, "Height": im.height, "Brightness": g.mean(), "Contrast": g.std()})
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Loading model...")
def get_model(name):
    import tensorflow as tf
    return tf.keras.models.load_model(config.MODELS_DIR / f"{name}.h5", compile=False)


def available_models():
    return sorted(p.stem for p in config.MODELS_DIR.glob("*.h5"))


def metrics_df(metrics):
    return pd.DataFrame([{
        "key": k, "Model": pretty(k), "Accuracy": v["accuracy"], "Precision": v["precision"],
        "Recall": v["recall"], "F1": v["f1"], "Latency (ms/img)": v["latency_ms"],
        "Params (M)": v["params"] / 1e6, "Size (MB)": v["size_mb"],
    } for k, v in metrics.items()])


def style_fig(fig, height=380):
    fig.update_layout(height=height, colorway=COLORWAY, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(family="IBM Plex Sans", color=INK), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def empty_state(msg, cmd):
    st.info(msg)
    st.code(cmd, language="bash")


# ------------------------------------------------------------------ pages
def page_overview():
    metrics, df = load_metrics(), metrics_df(load_metrics())
    counts = split_counts()
    st.markdown('<div class="masthead"><h1>NeuroScan</h1>'
                "<p>Brain MRI tumor classification: compare models, inspect the data, and run a live diagnosis.</p></div>",
                unsafe_allow_html=True)
    c = st.columns(4)
    if df.empty:
        c[0].metric("Models trained", 0)
    else:
        best = df.sort_values("F1", ascending=False).iloc[0]
        c[0].metric("Top model (macro F1)", best["Model"])
        c[1].metric("Test accuracy", f"{best['Accuracy']:.2%}")
        c[2].metric("Macro F1", f"{best['F1']:.3f}")
        c[3].metric("Models trained", len(df))
    if df.empty:
        empty_state("No trained models yet. Download the dataset into data/, then train:",
                    "python data_utils.py   # check the dataset\npython train.py        # trains all 5 models")

    st.subheader("Tumor categories")
    names = load_class_names()
    if names:
        cols = st.columns(len(names))
        for col, n in zip(cols, names):
            label, desc, urgency, color = class_info(n)
            total = int(counts[counts.Class == n].Images.sum()) if not counts.empty else 0
            col.markdown(f'<div class="panel"><span class="tag" style="background:{color}">{urgency} priority</span>'
                         f"<h4 style='margin:.5rem 0 .2rem'>{label}</h4><small>{desc}</small><br>"
                         f"<small><b>{total:,}</b> images in dataset</small></div>", unsafe_allow_html=True)
    st.subheader("How the pipeline works")
    steps = ["Explore data", "Normalize + resize", "Augment", "Train CNN + 4 pretrained", "Evaluate", "Deploy"]
    for col, s in zip(st.columns(len(steps)), steps):
        col.markdown(f'<div class="panel" style="text-align:center"><b>{s}</b></div>', unsafe_allow_html=True)


def page_dataset():
    st.header("Dataset explorer")
    counts = split_counts()
    if counts.empty:
        return empty_state("Dataset not found. Place it as data/train, data/test (and optionally data/valid).",
                           "python data_utils.py")
    left, right = st.columns([2, 1])
    splits = left.multiselect("Splits", counts.Split.unique().tolist(), default=counts.Split.unique().tolist())
    chart = right.radio("Chart", ["Bar", "Pie", "Treemap"], horizontal=True)
    view = counts[counts.Split.isin(splits)]
    if view.empty:
        return st.warning("Select at least one split.")
    if chart == "Bar":
        fig = px.bar(view, x="Class", y="Images", color="Split", barmode="group", text_auto=True)
    elif chart == "Pie":
        fig = px.pie(view.groupby("Class", as_index=False).Images.sum(), names="Class", values="Images", hole=0.45)
    else:
        fig = px.treemap(view, path=["Split", "Class"], values="Images")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    per_class = view.groupby("Class").Images.sum()
    ratio = per_class.max() / max(per_class.min(), 1)
    m = st.columns(3)
    m[0].metric("Total images", f"{int(per_class.sum()):,}")
    m[1].metric("Largest / smallest class", f"{ratio:.2f}x")
    m[2].metric("Imbalance", "Low" if ratio < 1.5 else "Moderate" if ratio < 3 else "High")
    if ratio >= 1.5:
        st.caption("Training uses inverse-frequency class weights to compensate for this imbalance.")

    st.subheader("Sample gallery")
    dirs = data_utils.get_split_dirs()
    g = st.columns([1, 1, 1, 1])
    split = g[0].selectbox("Split", [s for s, p in dirs.items() if p])
    cls = g[1].selectbox("Class", sorted(counts.Class.unique()))
    n = g[2].slider("Images", 4, 12, 8, 4)
    if g[3].button("Shuffle"):
        st.session_state.seed = random.random()
    files = data_utils.list_images(dirs[split] / cls) if (dirs[split] / cls).exists() else []
    if files:
        picks = random.Random(st.session_state.get("seed", 0)).sample(files, min(n, len(files)))
        for i, col in enumerate(st.columns(4) * (len(picks) // 4 + 1)):
            if i < len(picks):
                col.image(Image.open(picks[i]), caption=picks[i].name, use_container_width=True)

    st.subheader("Resolution and intensity check")
    train = dirs["train"] or next(p for p in dirs.values() if p)
    all_files = [str(p) for p in data_utils.list_images(train)]
    k = st.slider("Images to sample", 50, min(600, max(50, len(all_files))), 200, 50)
    stats = image_stats(tuple(random.Random(0).sample(all_files, min(k, len(all_files)))))
    a, b = st.columns(2)
    sizes = stats.groupby(["Width", "Height"]).size().reset_index(name="Count")
    a.plotly_chart(style_fig(px.scatter(sizes, x="Width", y="Height", size="Count",
                                        title=f"{len(sizes)} distinct resolutions (resized to {config.IMG_SIZE[0]}px)"), 340),
                   use_container_width=True)
    b.plotly_chart(style_fig(px.histogram(stats, x="Brightness", nbins=30, title="Mean brightness distribution"), 340),
                   use_container_width=True)


def page_training():
    st.header("Training insights")
    names = [n for n in config.MODEL_NAMES if load_history(n)]
    if not names:
        return empty_state("No training histories found. Run:", "python train.py")
    sel = st.selectbox("Model", names, format_func=pretty)
    smooth = st.slider("Smoothing", 0.0, 0.9, 0.0, 0.05)
    data = load_history(sel)
    h, switch = data["history"], data["finetune_start_epoch"]

    def ema(v):
        out, s = [], None
        for x in v:
            s = x if s is None else smooth * s + (1 - smooth) * x
            out.append(s)
        return out

    ep = list(range(1, len(h["loss"]) + 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))
    for col, (tr, va) in enumerate([("accuracy", "val_accuracy"), ("loss", "val_loss")], start=1):
        fig.add_scatter(x=ep, y=ema(h[tr]), name="Train", line=dict(color=TEAL), showlegend=col == 1, row=1, col=col)
        fig.add_scatter(x=ep, y=ema(h[va]), name="Validation", line=dict(color=AMBER), showlegend=col == 1, row=1, col=col)
        if switch:
            fig.add_vline(x=switch + 0.5, line_dash="dot", line_color="#7C9CB2", row=1, col=col)
    fig.update_xaxes(title_text="Epoch")
    st.plotly_chart(style_fig(fig, 400), use_container_width=True)
    if switch:
        st.caption("Dotted line: fine-tuning of the top backbone layers starts here.")

    best = int(np.argmin(h["val_loss"]))
    m = st.columns(4)
    m[0].metric("Best epoch (val loss)", best + 1)
    m[1].metric("Val accuracy there", f"{h['val_accuracy'][best]:.2%}")
    m[2].metric("Train - val gap", f"{h['accuracy'][best] - h['val_accuracy'][best]:+.2%}")
    m[3].metric("Epochs run", len(ep))

    if st.toggle("Overlay validation accuracy of all models"):
        fig = go.Figure()
        for n in names:
            v = load_history(n)["history"]["val_accuracy"]
            fig.add_scatter(x=list(range(1, len(v) + 1)), y=v, name=pretty(n))
        fig.update_layout(xaxis_title="Epoch", yaxis_title="Validation accuracy")
        st.plotly_chart(style_fig(fig), use_container_width=True)


def page_comparison():
    st.header("Model comparison")
    metrics = load_metrics()
    if not metrics:
        return empty_state("No evaluation results found. Run:", "python train.py   # or: python evaluate.py")
    df = metrics_df(metrics)
    score_cols = ["Accuracy", "Precision", "Recall", "F1"]
    st.dataframe(df.drop(columns="key").sort_values("F1", ascending=False).style
                 .format({c: "{:.3f}" for c in score_cols} | {"Latency (ms/img)": "{:.1f}", "Params (M)": "{:.1f}", "Size (MB)": "{:.0f}"})
                 .background_gradient(subset=score_cols, cmap="Blues"), use_container_width=True, hide_index=True)
    st.download_button("Download comparison (CSV)", df.drop(columns="key").to_csv(index=False), "model_comparison.csv")

    a, b = st.columns(2)
    chosen = a.multiselect("Models", df.Model.tolist(), default=df.Model.tolist())
    shown = df[df.Model.isin(chosen)]
    long = shown.melt(id_vars="Model", value_vars=score_cols, var_name="Metric", value_name="Score")
    a.plotly_chart(style_fig(px.bar(long, x="Metric", y="Score", color="Model", barmode="group").update_yaxes(range=[0, 1.02])),
                   use_container_width=True)
    radar = go.Figure()
    for _, r in shown.iterrows():
        radar.add_trace(go.Scatterpolar(r=[r[c] for c in score_cols + [score_cols[0]]],
                                        theta=score_cols + [score_cols[0]], name=r.Model, fill="toself", opacity=0.45))
    radar.update_layout(polar=dict(radialaxis=dict(range=[max(0, shown[score_cols].min().min() - 0.05), 1])))
    b.plotly_chart(style_fig(radar), use_container_width=True)
    st.plotly_chart(style_fig(px.scatter(shown, x="Latency (ms/img)", y="F1", size="Params (M)", color="Model",
                                         text="Model", title="Accuracy vs speed (bubble = parameters)")),
                    use_container_width=True)

    st.subheader("Which model should we deploy?")
    c = st.columns(2)
    prio = c[0].radio("Priority", ["Accuracy", "Speed", "Size", "Balanced"], horizontal=True)
    max_lat = c[1].slider("Max latency (ms/img)", float(df["Latency (ms/img)"].min()),
                          float(df["Latency (ms/img)"].max()), float(df["Latency (ms/img)"].max()))
    pool = df[df["Latency (ms/img)"] <= max_lat].copy()
    norm = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)  # noqa: E731
    pool["score"] = {"Accuracy": pool.F1, "Speed": -pool["Latency (ms/img)"], "Size": -pool["Size (MB)"],
                     "Balanced": 0.6 * norm(pool.F1) + 0.2 * norm(-pool["Latency (ms/img)"]) + 0.2 * norm(-pool["Size (MB)"])}[prio]
    if pool.empty:
        st.warning("No model meets that latency limit.")
    else:
        top = pool.sort_values("score", ascending=False).iloc[0]
        st.success(f"Recommended: **{top.Model}**  -  F1 {top.F1:.3f}, {top['Latency (ms/img)']:.1f} ms/img, {top['Size (MB)']:.0f} MB")

    st.subheader("Errors in detail")
    sel = st.selectbox("Model", df.key.tolist(), format_func=pretty)
    m = metrics[sel]
    norm_cm = st.toggle("Normalize by actual class", True)
    cm = np.array(m["confusion_matrix"], dtype=float)
    if norm_cm:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig = px.imshow(cm, x=m["class_names"], y=m["class_names"], text_auto=".2f" if norm_cm else "d",
                    color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual", color=""))
    pc = pd.DataFrame(m["per_class"]).T.reset_index(names="Class").melt(
        id_vars="Class", value_vars=["precision", "recall", "f1"], var_name="Metric", value_name="Score")
    x, y = st.columns(2)
    x.plotly_chart(style_fig(fig), use_container_width=True)
    y.plotly_chart(style_fig(px.bar(pc, x="Class", y="Score", color="Metric", barmode="group").update_yaxes(range=[0, 1.02])),
                   use_container_width=True)
    st.caption("For screening, watch recall on tumor classes: a missed tumor costs more than a false alarm.")


def preprocess(img):
    return np.asarray(img.convert("RGB").resize(config.IMG_SIZE, Image.BILINEAR), dtype="float32") / 255.0


def page_predict():
    st.header("Live diagnosis")
    models, names = available_models(), load_class_names()
    if not models:
        return empty_state("No trained models in saved_models/. Train first:", "python train.py")
    metrics = load_metrics()
    best = max(models, key=lambda m: metrics.get(m, {}).get("f1", 0))

    c = st.columns([2, 1, 1, 1])
    options = models + (["ensemble"] if len(models) > 1 else [])
    choice = c[0].selectbox("Model", options, index=options.index(best),
                            format_func=lambda n: "Ensemble (average of all models)" if n == "ensemble" else pretty(n))
    thr = c[1].slider("Review below", 0.5, 0.99, 0.85, 0.01, help="Predictions under this confidence are flagged for a specialist.")
    cam_on = c[2].toggle("Grad-CAM", True)
    alpha = c[3].slider("Overlay", 0.1, 0.9, 0.45, 0.05, disabled=not cam_on)

    source = st.radio("Image source", ["Upload MRI scans", "Random test-set image"], horizontal=True)
    images = []
    if source == "Upload MRI scans":
        for f in st.file_uploader("MRI images", type=["jpg", "jpeg", "png"], accept_multiple_files=True) or []:
            images.append((f.name, Image.open(f), None))
    else:
        test_dir = data_utils.get_split_dirs()["test"]
        if test_dir is None:
            st.warning("No test folder found in data/.")
        else:
            cls = st.selectbox("Class", ["any"] + names)
            if st.button("Pick a random image") or "sample" not in st.session_state:
                pool = data_utils.list_images(test_dir / cls) if cls != "any" else data_utils.list_images(test_dir)
                st.session_state.sample = random.choice(pool) if pool else None
            s = st.session_state.get("sample")
            if s:
                images.append((f"{s.parent.name}/{s.name}", Image.open(s), s.parent.name))
    if not images:
        return st.info("Add an MRI image to see the predicted tumor type, confidence and the regions the model used.")

    log = st.session_state.setdefault("log", {})
    tabs = st.tabs([n[0][:22] for n in images]) if len(images) > 1 else [st.container()]
    for tab, (fname, img, truth) in zip(tabs, images):
        with tab:
            arr = preprocess(img)[None]
            per_model = {m: get_model(m).predict(arr, verbose=0)[0] for m in models} if choice == "ensemble" else \
                        {choice: get_model(choice).predict(arr, verbose=0)[0]}
            probs = np.mean(list(per_model.values()), axis=0)
            labels = names if len(names) == len(probs) else [f"class_{i}" for i in range(len(probs))]
            top = int(probs.argmax())
            label, desc, urgency, color = class_info(labels[top])
            conf = float(probs[top])

            l, m, r = st.columns([1, 1, 1.1])
            l.image(img, caption=fname, use_container_width=True)
            if cam_on:
                from gradcam import make_gradcam, overlay
                cam, _, _ = make_gradcam(get_model(best if choice == "ensemble" else choice), arr, top)
                m.image(overlay(img, cam, alpha), caption="Grad-CAM: regions driving the prediction", use_container_width=True)
            r.markdown(f'<div class="verdict" style="background:{color}">Predicted<b>{label}</b>'
                       f"{conf:.1%} confidence</div>", unsafe_allow_html=True)
            r.write(desc)
            if truth is not None:  # test-set image: the folder name is the ground truth
                if class_key(truth) == class_key(labels[top]):
                    r.success(f"Actual label: {class_info(truth)[0]}. The prediction is correct.")
                else:
                    r.error(f"Actual label: {class_info(truth)[0]}. The prediction is WRONG.")
            if conf < thr:
                r.warning("Low confidence. Send this scan for specialist review.")
            elif labels[top] != "" and class_key(labels[top]) != "notumor":
                r.error(f"Tumor pattern detected ({urgency} priority). Flag for radiologist review.")
            else:
                r.success("No tumor pattern detected.")

            g, p = st.columns([1, 2])
            g.plotly_chart(style_fig(go.Figure(go.Indicator(mode="gauge+number", value=conf * 100,
                           number={"suffix": "%"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                                                          "threshold": {"line": {"color": INK, "width": 3}, "value": thr * 100}})), 250),
                           use_container_width=True)
            p.plotly_chart(style_fig(px.bar(x=probs, y=[class_info(x)[0] for x in labels], orientation="h",
                                            labels={"x": "Probability", "y": ""}).update_xaxes(range=[0, 1]), 250),
                           use_container_width=True)

            if len(models) > 1:
                with st.expander("What does each model say?"):
                    rows = []
                    for mname in models:
                        pr = per_model.get(mname)
                        if pr is None:
                            pr = get_model(mname).predict(arr, verbose=0)[0]
                        rows.append({"Model": pretty(mname), "Prediction": class_info(labels[int(pr.argmax())])[0],
                                     "Confidence": f"{pr.max():.1%}"})
                    tbl = pd.DataFrame(rows)
                    st.dataframe(tbl, hide_index=True, use_container_width=True)
                    st.caption("All models agree." if tbl.Prediction.nunique() == 1 else "Models disagree: treat as uncertain.")
            log[(fname, choice)] = {"Time": datetime.now().strftime("%H:%M:%S"), "Image": fname, "Model": pretty(choice),
                                    "Actual": class_info(truth)[0] if truth else "-", "Prediction": label, "Confidence": round(conf, 4), "Needs review": conf < thr}

    if log:
        st.subheader("Session log")
        out = pd.DataFrame(log.values())
        st.dataframe(out, hide_index=True, use_container_width=True)
        a, b = st.columns([1, 5])
        a.download_button("Download CSV", out.to_csv(index=False), "predictions.csv")
        if b.button("Clear log"):
            st.session_state.log = {}
            st.rerun()


def page_about():
    st.header("About")
    st.markdown("""
**Scope.** Classifies brain MRI slices by tumor type with a custom CNN and four ImageNet models
(ResNet50, MobileNetV2, InceptionV3, EfficientNetB0), fine-tuned in two phases.

**Deliverables.** Trained `.h5` models, this dashboard, training/evaluation scripts, model comparison, README.

**Important.** This is a decision-support prototype for research and demonstration. It is not a
certified medical device and must not replace a radiologist's judgment.
""")


PAGES = {"Overview": page_overview, "Dataset explorer": page_dataset, "Training insights": page_training,
         "Model comparison": page_comparison, "Live diagnosis": page_predict, "About": page_about}

st.sidebar.title("NeuroScan")
page = st.sidebar.radio("Go to", list(PAGES), label_visibility="collapsed")
st.sidebar.caption(f"{len(available_models())} trained model(s) found")
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()
PAGES[page]()