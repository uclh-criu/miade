## MiADE Training Data Dashboard

A helper streamlit app for data visualisation, training MedCAT models, and demo-ing MiADE.

To run locally:
Create a virtualenv, then:
```bash
pip install -r requirements.txt
streamlit run app.py
```
In Docker:
```bash
 docker build -t streamlit .
 docker run --name miade-data-app -p 8501:8501 streamlit
```