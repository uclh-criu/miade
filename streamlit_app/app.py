import os

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode
from st_aggrid.shared import GridUpdateMode
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from dotenv import load_dotenv, find_dotenv

from utils import MiADE_MetaCAT

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


MIN_HEIGHT = 27
MAX_HEIGHT = 800
ROW_HEIGHT = 35

TRAIN_DATA_DF = pd.read_csv(os.getenv("VIZ_DATA_PATH"))

SYNTH_DATA_OPTIONS = [f for f in os.listdir(os.getenv("SYNTH_CSV_DIR")) if ".csv" in f]
TRAIN_JSON_OPTIONS = [f for f in os.listdir(os.getenv("TRAIN_JSON_DIR")) if ".json" in f]
TEST_JSON_OPTIONS = [f for f in os.listdir(os.getenv("TEST_JSON_DIR")) if ".json" in f]
MODEL_OPTIONS = ["/".join(f[0].split("/")[-2:]) for f in os.walk(os.getenv("MODELS_DIR"))
                 if 'meta_' in f[0].split("/")[-1] and ".ipynb_checkpoints" not in f[0]]

st.set_page_config(
    layout="wide", page_icon="🖱️", page_title="Interactive train app"
)
st.title("🖱️ MiADE Training Dashboard")
st.write(
    """Miade MedCAT training dashboard"""
)

model_path = st.sidebar.selectbox("Select MetaCAT model path", MODEL_OPTIONS)
model_path = os.path.join(os.getenv("MODELS_DIR"), "/".join(model_path.split("/")[-2:]))

try:
    mc = MiADE_MetaCAT.load(model_path)
    model_name = mc.config.general["category_name"]
except Exception as e:
    st.error(f"Error loading model: {e}")
    mc = None

st.sidebar.subheader("Set training parameters")
cntx_left = st.sidebar.number_input("cntx_left", 5, step=1)
cntx_right = st.sidebar.number_input("cntx_right", 5, step=1)
n_epochs = st.sidebar.number_input("n_epochs", value=50, step=1, min_value=1)
is_replace_center = st.sidebar.checkbox("Replace centre token?", value=True)
replace_center = None
if is_replace_center:
    replace_center = st.sidebar.text_input("replace_center", "disease")

# TODO
st.sidebar.subheader("Create MedCAT modelpack")
st.sidebar.selectbox("Select MedCAT modelpack to package with:", ["miade_example_model"])
st.sidebar.button("Save")

tab1, tab2, tab3, tab4 = st.tabs(["Data", "Train", "Test", "Try"])


def present_confusion_matrix(model, data):
    data_name = Path(data).stem
    model_name = model.config.general['category_name']
    title = f"{model_name} evaluated against\n{data_name}"

    evaluation = model.eval(data)

    cm = evaluation["confusion matrix"].values
    label_names = [name.split()[-1] for name in list(evaluation["confusion matrix"].columns)]
    stats_text = "\n\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
        evaluation['precision'], evaluation['recall'], evaluation['f1'])

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
        fmt='',
    )
    conf.set(xlabel='True' + stats_text, ylabel='Predicted', title=title)
    st.write(evaluation)
    return plt


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


with tab1:
    # hard coded
    data_path = st.selectbox("Select dataset to interact with:", ["MiADE train data (400)"])
    is_load = st.checkbox("Show interactive table")
    if is_load:
        selection = aggrid_interactive_table(df=TRAIN_DATA_DF)
        if selection:
            st.write("You selected:")
            st.json(selection["selected_rows"])

with tab2:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Adjust** the sliders to vary the amount of synthetic data "
                    " you want to include in the training data in addition to your annotations:")
        train_json_path = st.selectbox("Select annotated training data", TRAIN_JSON_OPTIONS)
        train_json_path = os.path.join(os.getenv("TRAIN_JSON_DIR"), train_json_path)

        synth_csv_path = st.selectbox("Select synthetic data file:", SYNTH_DATA_OPTIONS)
        synth_csv_path = os.path.join(os.getenv("SYNTH_CSV_DIR"), synth_csv_path)
        synth_df = pd.read_csv(synth_csv_path)
        if mc is not None:
            category_labels = list(mc.config.general["category_value2id"].keys())
            label_counts = TRAIN_DATA_DF[model_name].value_counts().to_dict()
            synthetic_label_counts = synth_df[model_name].value_counts().to_dict()
            max_class = max(label_counts.values())

            assert len(category_labels) > 0
            assert set(category_labels).issubset(list(synthetic_label_counts.keys()))

            synth_add_dict = {}
            for i in range(len(category_labels)):
                synth_add_dict[category_labels[i]] = st.slider(category_labels[i] + " (synthetic)", min_value=0,
                                                               max_value=synthetic_label_counts[category_labels[i]],
                                                               value=max_class - label_counts[category_labels[i]])
    with col2:
        st.markdown("**Visualise** the ratio of real and synthetic in your overall training set:")
        if mc is not None:
            chart_data = pd.DataFrame(
                {"real": [label_counts[category_labels[i]] for i in range(len(category_labels))],
                 "synthetic": synth_add_dict.values()},
                index=category_labels)

            st.bar_chart(chart_data, height=500)

            train_df = pd.concat([synth_df[synth_df[model_name] == label][:synth_add_dict[label]]
                                  for label in category_labels],
                                 ignore_index=True)

    with col3:
        st.markdown("**Inspect** the sample of synthetic data selected:")
        if mc is not None:
            st.dataframe(train_df[["text", model_name]], height=500)

    if st.button('Train'):
        if mc is not None:
            with st.spinner("Training MetaCAT..."):
                date_id = datetime.now().strftime("%y%m%d%H%M%S")
                save_dir = "/".join(model_path.split("/")[:-2]) + "/" + date_id
                data_save_name = save_dir + "/synth_train_df.csv"
                model_save_name = save_dir + "/meta_" + model_name

                mc.config.general["cntx_left"] = cntx_left
                mc.config.general["cntx_right"] = cntx_right
                mc.config.train["nepochs"] = n_epochs
                mc.config.general["replace_center"] = replace_center
                mc.config.model["last_trained_on"] = date_id

                if len(train_df) == 0:
                    train_df = None

                with st.expander("Expand to see training logs"):
                    output = st.empty()
                    with st_capture(output.code):
                        report = mc.train(json_path=train_json_path,
                                          synthetic_data_df=train_df,
                                          save_dir_path=model_save_name)
            # save the generated train dataset
            if train_df is not None:
                train_df.to_csv(data_save_name)

            st.success(f"Done! Model saved at {model_save_name}")
            st.write("Training report:")
            st.write(report)
        else:
            st.error("No model loaded")

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        test_path = st.selectbox("Select a test set to evaluate your model against:", TEST_JSON_OPTIONS)
        test_path = os.path.join(os.getenv("TEST_JSON_DIR"), test_path)
        is_test = st.button("Test")
    with col2:
        cm = st.empty()
        if is_test:
            if mc is not None:
                plt = present_confusion_matrix(mc, test_path)
                cm.pyplot(plt)
            else:
                st.error("No model loaded")
        out = st.empty()

with tab4:
    # TODO
    st.text_area("Text")
    st.button("Submit")
