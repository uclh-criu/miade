# Welcome to the MiADE Documentation

![](assets/miade-logo.png)

MiADE (Medical information AI Data Extractor) is a set of tools for extracting formattable data from clinical notes stored in electronic health record systems (EHRs). Powered by Cogstack's [MedCAT](https://github.com/CogStack/MedCAT).

## Installing

```bash
pip install miade
```

You may also need to download these additional models to run MiADE:

[spaCy](https://spacy.io/models/en)
```bash
python -m spacy download en_core_web_md
```
[med7](https://huggingface.co/kormilitzin/en_core_med7_lg)
```bash
pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl
```

## License

MiADE is licensed under [Elastic License 2.0](https://www.elastic.co/licensing/elastic-license).

The Elastic License 2.0 is a flexible license that allows you to use, copy, distribute, make available, and prepare derivative works of the software, as long as you do not provide the software to others as a managed service or include it in a free software directory. For the full license text, see our [license page](https://github.com/uclh-criu/miade/blob/master/LICENCE.md).