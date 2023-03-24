import json
import os
import torch
import pandas as pd
from datetime import datetime

from typing import List, Dict, Optional

from medcat.meta_cat import MetaCAT
from medcat.tokenizers.meta_cat_tokenizers import TokenizerWrapperBase
from medcat.utils.meta_cat.data_utils import prepare_from_json, encode_category_values
from medcat.utils.meta_cat.ml_utils import train_model


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
            for ind, pair in enumerate(doc_text['offset_mapping']):
                if pair[0] <= start < pair[1]:
                    break

            _start = max(0, ind - cntx_left)
            _end = min(len(doc_text['input_ids']), ind + 1 + cntx_right)
            tkns = doc_text['input_ids'][_start:_end]
            cpos = cntx_left + min(0, ind - cntx_left)

            if replace_center is not None:
                if lowercase:
                    replace_center = replace_center.lower()
                for p_ind, pair in enumerate(doc_text['offset_mapping']):
                    if pair[0] <= start < pair[1]:
                        s_ind = p_ind
                    if pair[0] < end <= pair[1]:
                        e_ind = p_ind

                ln = e_ind - s_ind
                tkns = tkns[:cpos] + tokenizer(replace_center)['input_ids'] + tkns[cpos + ln + 1:]

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
            synthetic_data_df: Optional[pd.DataFrame] = None,
            save_dir_path: Optional[str] = None
    ) -> Dict:
        """
        synthetic_csv_path: MUST be a path to csv file with data [text, start, end, category_name] - output from
        miade synthetic data generator
        """

        g_config = self.config.general
        t_config = self.config.train

        # Load the medcattrainer export
        with open(json_path, 'r') as f:
            data_loaded: Dict = json.load(f)

        # Create directories if they don't exist
        if t_config['auto_save_model']:
            if save_dir_path is None:
                raise Exception("The `save_dir_path` argument is required if `aut_save_model` is `True` in the config")
            else:
                os.makedirs(save_dir_path, exist_ok=True)

        # Prepare the data
        assert self.tokenizer is not None
        data = prepare_from_json(data_loaded, g_config['cntx_left'], g_config['cntx_right'], self.tokenizer,
                                 cui_filter=t_config['cui_filter'],
                                 replace_center=g_config['replace_center'], prerequisites=t_config['prerequisites'],
                                 lowercase=g_config['lowercase'])

        # Check is the name there
        category_name = g_config['category_name']
        if category_name not in data:
            raise Exception(
                "The category name does not exist in this json file. You've provided '{}', while the possible options "
                "are: {}".format(
                    category_name, " | ".join(list(data.keys()))))

        data = data[category_name]

        if synthetic_data_df is not None:
            self.log.info(
                f"Training with additional {len(synthetic_data_df)} synthetic data points")
            synth_data = prepare_from_miade_csv(synthetic_data_df,
                                                cntx_left=g_config['cntx_left'],
                                                cntx_right=g_config['cntx_right'],
                                                tokenizer=self.tokenizer,
                                                replace_center=g_config['replace_center'],
                                                category_name=category_name,
                                                lowercase=g_config['lowercase'])
            synth_data = synth_data[category_name]
            # concat synth data to medcattrainer data
            data = data + synth_data

        category_value2id = g_config['category_value2id']
        if not category_value2id:
            # Encode the category values
            data, category_value2id = encode_category_values(data)
            g_config['category_value2id'] = category_value2id
        else:
            # We already have everything, just get the data
            data, _ = encode_category_values(data, existing_category_value2id=category_value2id)

        # Make sure the config number of classes is the same as the one found in the data
        if len(category_value2id) != self.config.model['nclasses']:
            self.log.warning(
                "The number of classes set in the config is not the same as the one found in the data: {} vs {}".format(
                    self.config.model['nclasses'], len(category_value2id)))
            self.log.warning("Auto-setting the nclasses value in config and rebuilding the model.")
            self.config.model['nclasses'] = len(category_value2id)
            self.model = self.get_model(embeddings=self.embeddings)

        report = train_model(self.model, data=data, config=self.config, save_dir_path=save_dir_path)

        # If autosave, then load the best model here
        if t_config['auto_save_model']:
            if save_dir_path is None:
                raise Exception("The `save_dir_path` argument is required if `aut_save_model` is `True` in the config")
            else:
                path = os.path.join(save_dir_path, 'model.dat')
                device = torch.device(g_config['device'])
                self.model.load_state_dict(torch.load(path, map_location=device))

                # Save everything now
                self.save(save_dir_path=save_dir_path)

        self.config.train['last_train_on'] = datetime.now().timestamp()
        return report
