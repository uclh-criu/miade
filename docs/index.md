# Welcome to the MiADE Documentation

![](assets/miade-logo.png)

MiADE (Medical information AI Data Extractor) is a set of tools for extracting formattable data from clinical notes stored in electronic health record systems (EHRs). Built with Cogstack's [MedCAT](https://github.com/CogStack/MedCAT) package.

## Installing

To install MiADE, you need to download the spacy base model and Med7 model first:

```bash
pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl
python -m spacy download en_core_web_md
```
Then, install MiADE:

```bash
pip install miade
```

## License

MiADE is licensed under Elastic License 2.0