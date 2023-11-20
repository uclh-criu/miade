import json
import os
import torch
import pandas as pd
import logging
from datetime import datetime

import math
import numpy as np
from typing import List, Optional, Dict, Tuple, Any
from torch import nn
from medcat.config_meta_cat import ConfigMetaCAT
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from medcat.meta_cat import MetaCAT
from medcat.tokenizers.meta_cat_tokenizers import TokenizerWrapperBase
from medcat.utils.meta_cat.data_utils import prepare_from_json, encode_category_values
from medcat.utils.meta_cat.ml_utils import train_model

logger = logging.getLogger("meta_cat")


# Hacky as hell, just for the dashboard, NOT permanent solution - will not merge with main branch
def create_batch_piped_data(data: List, start_ind: int, end_ind: int, device: torch.device, pad_id: int) -> Tuple:
    """Creates a batch given data and start/end that denote batch size, will also add
    padding and move to the right device.
    Args:
        data (List[List[int], int, Optional[int]]):
            Data in the format: [[<[input_ids]>, <cpos>, Optional[int]], ...], the third column is optional
            and represents the output label
        start_ind (int):
            Start index of this batch
        end_ind (int):
            End index of this batch
        device (torch.device):
            Where to move the data
        pad_id (int):
            Padding index
    Returns:
        x ():
            Same as data, but subsetted and as a tensor
        cpos ():
            Center positions for the data
    """
    max_seq_len = max([len(x[0]) for x in data])
    x = [x[0][0:max_seq_len] + [pad_id] * max(0, max_seq_len - len(x[0])) for x in data[start_ind:end_ind]]
    cpos = [x[1] for x in data[start_ind:end_ind]]
    y = None
    if len(data[0]) == 3:
        # Means we have the y column
        y = torch.tensor([x[2] for x in data[start_ind:end_ind]], dtype=torch.long).to(device)

    x = torch.tensor(x, dtype=torch.long).to(device)
    cpos = torch.tensor(cpos, dtype=torch.long).to(device)

    return x, cpos, y


def print_report(epoch: int, running_loss: List, all_logits: List, y: Any, name: str = "Train") -> None:
    r"""Prints some basic stats during training
    Args:
        epoch
        running_loss
        all_logits
        y
        name
    """
    if all_logits:
        print(f"Epoch: {epoch} " + "*" * 50 + f"  {name}")
        print(classification_report(y, np.argmax(np.concatenate(all_logits, axis=0), axis=1)))


def eval_model(model: nn.Module, data: List, config: ConfigMetaCAT, tokenizer: TokenizerWrapperBase) -> Dict:
    """Evaluate a trained model on the provided data
    Args:
        model
        data
        config
    """
    device = torch.device(config.general["device"])  # Create a torch device
    batch_size_eval = config.general["batch_size_eval"]
    pad_id = config.model["padding_idx"]
    ignore_cpos = config.model["ignore_cpos"]
    class_weights = config.train["class_weights"]

    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)  # Set the criterion to Cross Entropy Loss
    else:
        criterion = nn.CrossEntropyLoss()  # Set the criterion to Cross Entropy Loss

    y_eval = [x[2] for x in data]
    num_batches = math.ceil(len(data) / batch_size_eval)
    running_loss = []
    all_logits = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i in range(num_batches):
            x, cpos, y = create_batch_piped_data(
                data, i * batch_size_eval, (i + 1) * batch_size_eval, device=device, pad_id=pad_id
            )
            logits = model(x, cpos, ignore_cpos=ignore_cpos)
            loss = criterion(logits, y)

            # Track loss and logits
            running_loss.append(loss.item())
            all_logits.append(logits.detach().cpu().numpy())

    print_report(0, running_loss, all_logits, y=y_eval, name="Eval")

    score_average = config.train["score_average"]
    predictions = np.argmax(np.concatenate(all_logits, axis=0), axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y_eval, predictions, average=score_average)

    labels = [name for (name, _) in sorted(config.general["category_value2id"].items(), key=lambda x: x[1])]
    confusion = pd.DataFrame(
        data=confusion_matrix(
            y_eval,
            predictions,
        ),
        columns=["true " + label for label in labels],
        index=["predicted " + label for label in labels],
    )

    examples: Dict = {"FP": {}, "FN": {}, "TP": {}}
    id2category_value = {v: k for k, v in config.general["category_value2id"].items()}
    for i, p in enumerate(predictions):
        y = id2category_value[y_eval[i]]
        p = id2category_value[p]
        c = data[i][1]
        tkns = data[i][0]
        assert tokenizer.hf_tokenizers is not None
        text = (
            tokenizer.hf_tokenizers.decode(tkns[0:c])
            + " <<"
            + tokenizer.hf_tokenizers.decode(tkns[c : c + 1]).strip()
            + ">> "
            + tokenizer.hf_tokenizers.decode(tkns[c + 1 :])
        )
        info = "Predicted: {}, True: {}".format(p, y)
        if p != y:
            # We made a mistake
            examples["FN"][y] = examples["FN"].get(y, []) + [(info, text)]
            examples["FP"][p] = examples["FP"].get(p, []) + [(info, text)]
        else:
            examples["TP"][y] = examples["TP"].get(y, []) + [(info, text)]

    return {"precision": precision, "recall": recall, "f1": f1, "examples": examples, "confusion matrix": confusion}


def prepare_from_miade_csv(
    data: pd.DataFrame,
    category_name: str,
    cntx_left: int,
    cntx_right: int,
    tokenizer: TokenizerWrapperBase,
    replace_center: str = None,
    lowercase: bool = True,
) -> Dict:
    out_data: Dict = {}

    for i in range(len(data)):
        text = data.text.values[i]

        if len(text) > 0:
            doc_text = tokenizer(text)

            start = data.start.values[i]
            end = data.end.values[i]

            # Get the index of the center token
            ind = 0
            for ind, pair in enumerate(doc_text["offset_mapping"]):
                if pair[0] <= start < pair[1]:
                    break

            _start = max(0, ind - cntx_left)
            _end = min(len(doc_text["input_ids"]), ind + 1 + cntx_right)
            tkns = doc_text["input_ids"][_start:_end]
            cpos = cntx_left + min(0, ind - cntx_left)

            if replace_center is not None:
                if lowercase:
                    replace_center = replace_center.lower()
                for p_ind, pair in enumerate(doc_text["offset_mapping"]):
                    if pair[0] <= start < pair[1]:
                        s_ind = p_ind
                    if pair[0] < end <= pair[1]:
                        e_ind = p_ind

                ln = e_ind - s_ind
                tkns = tkns[:cpos] + tokenizer(replace_center)["input_ids"] + tkns[cpos + ln + 1 :]

            value = data[category_name].values[i]
            sample = [tkns, cpos, value]

            if category_name in out_data:
                out_data[category_name].append(sample)
            else:
                out_data[category_name] = [sample]

    return out_data


class MiADE_MetaCAT(MetaCAT):
    """Overrides MetaCAT train function"""

    def train(
        self,
        json_path: str,
        synthetic_csv_path: Optional[str] = None,
        save_dir_path: Optional[str] = None,
    ) -> Dict:
        """
        synthetic_csv_path: MUST be a path to csv file with data [text, start, end, category_name] - output from
        miade synthetic data generator
        """

        g_config = self.config.general
        t_config = self.config.train

        # Load the medcattrainer export
        with open(json_path, "r") as f:
            data_loaded: Dict = json.load(f)

        # Create directories if they don't exist
        if t_config["auto_save_model"]:
            if save_dir_path is None:
                raise Exception("The `save_dir_path` argument is required if `aut_save_model` is `True` in the config")
            else:
                os.makedirs(save_dir_path, exist_ok=True)

        # Prepare the data
        assert self.tokenizer is not None
        data = prepare_from_json(
            data_loaded,
            g_config["cntx_left"],
            g_config["cntx_right"],
            self.tokenizer,
            cui_filter=t_config["cui_filter"],
            replace_center=g_config["replace_center"],
            prerequisites=t_config["prerequisites"],
            lowercase=g_config["lowercase"],
        )

        # Check is the name there
        category_name = g_config["category_name"]
        if category_name not in data:
            raise Exception(
                "The category name does not exist in this json file. You've provided '{}', while the possible options "
                "are: {}".format(category_name, " | ".join(list(data.keys())))
            )

        data = data[category_name]

        if synthetic_csv_path is not None:
            synth_data_loaded = pd.read_csv(synthetic_csv_path)
            logger.info(
                f"Training with additional {len(synth_data_loaded)} synthetic data points from {synthetic_csv_path}"
            )
            synth_data = prepare_from_miade_csv(
                synth_data_loaded,
                cntx_left=g_config["cntx_left"],
                cntx_right=g_config["cntx_right"],
                tokenizer=self.tokenizer,
                replace_center=g_config["replace_center"],
                category_name=category_name,
                lowercase=g_config["lowercase"],
            )
            synth_data = synth_data[category_name]
            # concat synth data to medcattrainer data
            data = data + synth_data

        category_value2id = g_config["category_value2id"]
        if not category_value2id:
            # Encode the category values
            data, category_value2id = encode_category_values(data)
            g_config["category_value2id"] = category_value2id
        else:
            # We already have everything, just get the data
            data, _ = encode_category_values(data, existing_category_value2id=category_value2id)

        # Make sure the config number of classes is the same as the one found in the data
        if len(category_value2id) != self.config.model["nclasses"]:
            logger.warning(
                "The number of classes set in the config is not the same as the one found in the data: {} vs {}".format(
                    self.config.model["nclasses"], len(category_value2id)
                )
            )
            logger.warning("Auto-setting the nclasses value in config and rebuilding the model.")
            self.config.model["nclasses"] = len(category_value2id)
            self.model = self.get_model(embeddings=self.embeddings)

        report = train_model(self.model, data=data, config=self.config, save_dir_path=save_dir_path)

        # If autosave, then load the best model here
        if t_config["auto_save_model"]:
            if save_dir_path is None:
                raise Exception("The `save_dir_path` argument is required if `aut_save_model` is `True` in the config")
            else:
                path = os.path.join(save_dir_path, "model.dat")
                device = torch.device(g_config["device"])
                self.model.load_state_dict(torch.load(path, map_location=device))

                # Save everything now
                self.save(save_dir_path=save_dir_path)

        self.config.train["last_train_on"] = datetime.now().timestamp()
        return report

    def eval(self, json_path: str) -> Dict:
        """Evaluate from json."""
        g_config = self.config.general
        t_config = self.config.train

        with open(json_path, "r") as f:
            data_loaded: Dict = json.load(f)

        # Prepare the data
        assert self.tokenizer is not None
        data = prepare_from_json(
            data_loaded,
            g_config["cntx_left"],
            g_config["cntx_right"],
            self.tokenizer,
            cui_filter=t_config["cui_filter"],
            replace_center=g_config["replace_center"],
            prerequisites=t_config["prerequisites"],
            lowercase=g_config["lowercase"],
        )

        # Check is the name there
        category_name = g_config["category_name"]
        if category_name not in data:
            raise Exception("The category name does not exist in this json file.")

        data = data[category_name]

        # We already have everything, just get the data
        category_value2id = g_config["category_value2id"]
        data, _ = encode_category_values(data, existing_category_value2id=category_value2id)

        # Run evaluation
        assert self.tokenizer is not None
        result = eval_model(self.model, data, config=self.config, tokenizer=self.tokenizer)

        return result
