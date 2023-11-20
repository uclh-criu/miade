# ruff: noqa: F811

import os
import json
from time import sleep

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from typing import List

from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode
from st_aggrid.shared import GridUpdateMode
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from dotenv import load_dotenv, find_dotenv
from spacy_streamlit import visualize_ner

from medcat.cat import CAT
from miade.utils.miade_meta_cat import MiADE_MetaCAT
from utils import (
    load_documents,
    load_annotations,
    get_valid_annotations,
    get_probs_meta_classes_data,
    get_meds_meta_classes_data,
)

load_dotenv(find_dotenv())


@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write
        yield


@st.cache_data
def load_csv_data(csv_path):
    return pd.read_csv(csv_path)


@st.cache_data
def get_label_counts(name, train, synth):
    real_counts = {}
    synthetic_counts = {}
    real_labels = train.get(name)
    synthetic_labels = synth.get(name)
    if real_labels is not None:
        real_counts = real_labels.value_counts().to_dict()
    if synthetic_labels is not None:
        synthetic_counts = synthetic_labels.value_counts().to_dict()
    return real_counts, synthetic_counts


@st.cache_data
def get_chart_data(labels, label_count, synth_add_count):
    return pd.DataFrame(
        {"real": [label_count.get(labels[i], 0) for i in range(len(labels))], "synthetic": synth_add_count.values()},
        index=category_labels,
    )


@st.cache_data
def make_train_data(synth_df, name, labels, synth_add_count):
    return pd.concat(
        [synth_df[synth_df[name] == label][: synth_add_count[label]] for label in labels], ignore_index=True
    )


@st.cache_resource
def load_metacat_model(path):
    try:
        model = MiADE_MetaCAT.load(path)
        name = model.config.general["category_name"]
        sleep(0.5)
        st.sidebar.success(f"{name} model loaded from {model_path}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None
        name = None
    return model, name


@st.cache_resource
def load_medcat_model(path):
    try:
        model = CAT.load_model_pack(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None
    return model


MIN_HEIGHT = 27
MAX_HEIGHT = 800
ROW_HEIGHT = 35

SYNTH_DATA_OPTIONS = [f for f in os.listdir(os.getenv("SYNTH_CSV_DIR")) if ".csv" in f]
TRAIN_JSON_OPTIONS = [f for f in os.listdir(os.getenv("TRAIN_JSON_DIR")) if ".json" in f]
TEST_JSON_OPTIONS = [f for f in os.listdir(os.getenv("TEST_JSON_DIR")) if ".json" in f]

MEDCAT_OPTIONS = [f for f in os.listdir(os.getenv("SAVE_DIR")) if ".zip" in f]
MODEL_OPTIONS = [
    "/".join(f[0].split("/")[-2:])
    for f in os.walk(os.getenv("MODELS_DIR"))
    if "meta_" in f[0].split("/")[-1] and ".ipynb_checkpoints" not in f[0]
]

st.set_page_config(layout="wide", page_icon="🖱️", page_title="MiADE train app")
st.title("🖱️ MiADE Training Dashboard")
st.write("""Hello! Train, test, and experiment with MedCAT models used in MiADE""")


def present_confusion_matrix(model, data):
    data_name = Path(data).stem
    model_name = model.config.general["category_name"]
    title = f"{model_name} evaluated against\n{data_name}"

    evaluation = model.eval(data)

    cm = evaluation["confusion matrix"].values
    label_names = [name.split()[-1] for name in list(evaluation["confusion matrix"].columns)]
    stats_text = "\n\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
        evaluation["precision"], evaluation["recall"], evaluation["f1"]
    )

    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    box_labels = [f"{v1}\n{v2}".strip() for v1, v2 in zip(group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cm.shape[0], cm.shape[1])

    conf = sns.heatmap(
        cm,
        annot=box_labels,
        cmap="Blues",  # sns.light_palette("#03012d"),
        square=True,
        xticklabels=label_names,
        yticklabels=label_names,
        fmt="",
    )
    conf.set(xlabel="True" + stats_text, ylabel="Predicted", title=title)
    st.write(evaluation)
    return plt


def add_metacat_models(
    model: str,
    meta_cats_path: List,
):
    out_dir = os.getenv("SAVE_DIR", "./")
    cat = CAT.load_model_pack(str(model))

    meta_cats = []
    categories = []
    for metacat_path in meta_cats_path:
        metacat = MiADE_MetaCAT.load(metacat_path)
        meta_cats.append(metacat)
        categories.append(metacat.config.general["category_name"])

    cat_w_meta = CAT(cdb=cat.cdb, vocab=cat.vocab, config=cat.config, meta_cats=meta_cats)

    description = cat.config.version["description"] + " | Packaged with MetaCAT model(s) " + ", ".join(categories)
    cat.config.version["description"] = description
    save_name = Path(model).stem.rsplit("_", 3)[0] + "_w_meta_" + datetime.now().strftime("%b_%Y").lower()
    try:
        cat_w_meta.create_model_pack(save_dir_path=out_dir, model_pack_name=save_name)
        st.success("Saved MedCAT modelpack at " + out_dir + save_name + "_" + cat_w_meta.config.version["id"])
    except Exception as e:
        st.error(f"Error saving MedCAT model: {e}")


def aggrid_interactive_table(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.

    Args:
        df (pd.DataFrame]): Source dataframe

    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True, min_column_width=100
    )
    options.configure_selection(selection_mode="multiple", use_checkbox=True)
    options.configure_side_bar()

    options.configure_selection("single")
    selection = AgGrid(
        df,
        height=min(MIN_HEIGHT + len(df) * ROW_HEIGHT, MAX_HEIGHT),
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
        fit_columns_on_grid_load=True,
        gridOptions=options.build(),
        theme="streamlit",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection


# side bar
st.sidebar.subheader("Select model to train")
model_path = st.sidebar.selectbox("Select MetaCAT model path", MODEL_OPTIONS)
model_path = os.path.join(os.getenv("MODELS_DIR"), "/".join(model_path.split("/")[-2:]))
mc, model_name = load_metacat_model(model_path)

st.sidebar.subheader("Set training parameters")
cntx_left = st.sidebar.number_input("cntx_left", 5, step=1)
cntx_right = st.sidebar.number_input("cntx_right", 5, step=1)
n_epochs = st.sidebar.number_input("n_epochs", value=50, step=1, min_value=1)

is_replace_center = st.sidebar.checkbox("Replace centre token?", value=True)
replace_center = None
if is_replace_center:
    replace_center = st.sidebar.text_input("replace_center", "disease")

class_weights = st.sidebar.checkbox("Balance class weights? (Experimental)", value=False)

st.sidebar.subheader("Save with MedCAT modelpack")
with st.sidebar.form(key="model"):
    selected_models = st.multiselect("Select MetaCAT models:", MODEL_OPTIONS)
    metacat_paths = [os.path.join(os.getenv("MODELS_DIR"), "/".join(path.split("/")[-2:])) for path in selected_models]
    selected_medcat = st.selectbox("Select MedCAT modelpack to package with:", MEDCAT_OPTIONS)
    medcat_path = os.getenv("SAVE_DIR") + selected_medcat
    submit = st.form_submit_button(label="Save")
    if submit:
        add_metacat_models(medcat_path, metacat_paths)
        # update options probably a better way to do this
        MEDCAT_OPTIONS = [f for f in os.listdir(os.getenv("SAVE_DIR")) if ".zip" in f]


# load data
# train_data_df = load_csv_data(os.getenv("VIZ_DATA_PATH"))

tab1, tab2, tab3, tab4 = st.tabs(["Train", "Test", "Data", "Try"])

with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            "**Adjust** the sliders to vary the amount of synthetic data "
            " you want to include in the training data in addition to your annotations:"
        )
        train_json_path = st.selectbox("Select annotated training data", TRAIN_JSON_OPTIONS)

        train_csv = train_json_path.replace(".json", ".csv")
        train_csv_path = os.path.join(os.getenv("TRAIN_CSV_DIR"), train_csv)

        train_json_path = os.path.join(os.getenv("TRAIN_JSON_DIR"), train_json_path)

        if not os.path.exists(train_csv_path):
            with open(train_json_path) as file:
                train_data = json.load(file)
            train_text = load_documents(train_data)
            train_annotations = load_annotations(train_data)
            valid_train_ann = get_valid_annotations(train_annotations)
            if "problems" in train_json_path:
                train_data_df = get_probs_meta_classes_data(train_text, valid_train_ann)
            else:
                train_data_df = get_meds_meta_classes_data(train_text, valid_train_ann)
            train_data_df.to_csv(train_csv_path, index=False)
        else:
            train_data_df = load_csv_data(train_csv_path)

        synth_csv_path = st.selectbox("Select synthetic data file:", SYNTH_DATA_OPTIONS)
        synth_csv_path = os.path.join(os.getenv("SYNTH_CSV_DIR"), synth_csv_path)

        all_synth_df = load_csv_data(synth_csv_path)
        if mc is not None:
            category_labels = list(mc.config.general["category_value2id"].keys())
            real_label_counts, synthetic_label_counts = get_label_counts(model_name, train_data_df, all_synth_df)
            if real_label_counts:
                max_class = max(real_label_counts.values())
            else:
                max_class = 0

            assert len(category_labels) > 0
            assert set(category_labels).issubset(list(synthetic_label_counts.keys()))

            synth_add_dict = {}
            for i in range(len(category_labels)):
                synth_add_dict[category_labels[i]] = st.slider(
                    category_labels[i] + " (synthetic)",
                    min_value=0,
                    max_value=synthetic_label_counts.get(category_labels[i], 0),
                    value=max_class - real_label_counts.get(category_labels[i], 0),
                )
    with col2:
        st.markdown("**Visualise** the ratio of real and synthetic in your overall training set:")
        if mc is not None:
            chart_data = get_chart_data(category_labels, real_label_counts, synth_add_dict)
            st.bar_chart(chart_data, height=500)

            synth_train_df = make_train_data(all_synth_df, model_name, category_labels, synth_add_dict)

    with col3:
        st.markdown("**Inspect** the sample of synthetic data selected:")
        if mc is not None:
            st.dataframe(synth_train_df[["text", model_name]], height=500)

    if st.button("Train"):
        if mc is not None:
            with st.spinner("Training MetaCAT..."):
                date_id = datetime.now().strftime("%y%m%d%H%M%S")
                save_dir = "/".join(model_path.split("/")[:-2]) + "/" + date_id
                data_save_name = save_dir + "/synth_train_df.csv"
                model_save_name = save_dir + "/meta_" + model_name
                # make dir to save in
                os.makedirs(save_dir, exist_ok=True)
                # save the generated train dataset
                if len(synth_train_df) > 0:
                    synth_train_df.to_csv(data_save_name, index=False)
                else:
                    data_save_name = None

                if class_weights:
                    weights = []
                    for label in mc.config.general["category_value2id"].keys():
                        train_count = {}
                        synth_count = {}
                        train_data_column = train_data_df.get(model_name)
                        synth_data_column = synth_train_df.get(model_name)
                        if train_data_column is not None:
                            train_count = train_data_column.value_counts().to_dict()
                        if synth_data_column is not None:
                            synth_count = synth_data_column.value_counts().to_dict()
                        if not train_count:
                            train_length = 1  # min. num data in json
                        else:
                            train_length = len(train_data_df)
                        total_count = train_length + len(synth_train_df)
                        class_count = train_count.get(label, 0) + synth_count.get(label, 0)
                        weight = 1 - (class_count / total_count)
                        weights.append(weight)
                st.write(weights)
                mc.config.general["cntx_left"] = cntx_left
                mc.config.general["cntx_right"] = cntx_right
                mc.config.general["replace_center"] = replace_center
                mc.config.model["last_trained_on"] = date_id
                mc.config.train["nepochs"] = n_epochs
                mc.config.train["class_weights"] = weights

                with st.expander("Expand to see training logs"):
                    output = st.empty()
                    with st_capture(output.code):
                        report = mc.train(
                            json_path=train_json_path, synthetic_csv_path=data_save_name, save_dir_path=model_save_name
                        )

            st.success(f"Done! Model saved at {model_save_name}")
            st.write("Training report:")
            st.write(report)
            MODEL_OPTIONS = [
                "/".join(f[0].split("/")[-2:])
                for f in os.walk(os.getenv("MODELS_DIR"))
                if "meta_" in f[0].split("/")[-1] and ".ipynb_checkpoints" not in f[0]
            ]
        else:
            st.error("No model loaded")


with tab2:
    col1, col2 = st.columns(2)
    with col1:
        model_path = st.selectbox("Select MetaCAT model to evaluate", MODEL_OPTIONS)
        model_path = os.path.join(os.getenv("MODELS_DIR", "./"), "/".join(model_path.split("/")[-2:]))

        test_path = st.selectbox("Select a test set to evaluate your model against:", TEST_JSON_OPTIONS)
        test_path = os.path.join(os.getenv("TEST_JSON_DIR"), test_path)

        is_save = st.checkbox("Save confusion matrix figure to model directory")
        is_test = st.button("Test")
    with col2:
        cm = st.empty()
        if is_test:
            eval_mc, model_name = load_metacat_model(model_path)
            plt = present_confusion_matrix(eval_mc, test_path)
            cm.pyplot(plt)
            if is_save:
                try:
                    plt.savefig(model_path + "/confusion_matrix.png", format="png", bbox_inches="tight", dpi=200)
                except Exception as e:
                    st.error(f"Could not save image: {e}")

        out = st.empty()


with tab3:
    # hard coded
    data_path = st.selectbox("Select dataset to interact with:", ["MiADE train data (400)"])
    is_load = st.checkbox("Show table")
    if is_load:
        selection = aggrid_interactive_table(df=train_data_df)
        if selection:
            st.write("You selected:")
            st.json(selection["selected_rows"])


with tab4:
    col4, col5 = st.columns(2)
    with col4:
        st.markdown("Try it out!")
        selected_medcat = st.selectbox("Select a MedCAT modelpack to load:", MEDCAT_OPTIONS)
        medcat_path = os.getenv("SAVE_DIR") + selected_medcat
        if st.button("Load MedCAT model"):
            cat = load_medcat_model(medcat_path)
        text = st.text_area("Text")
        submit = st.button("Submit")
    with col5:
        if submit:
            cat = load_medcat_model(medcat_path)
            output = cat.get_entities(text)
            doc = cat(text)
            visualize_ner(doc, title=None, show_table=False, displacy_options={"colors": {"concept": "#F17156"}})
            st.write(output)
