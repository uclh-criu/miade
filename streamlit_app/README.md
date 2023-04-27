## MiADE Training Data Dashboard

A helper streamlit app for data visualisation, training MedCAT models, and demo-ing MiADE. NOTE: You must have the main miade package installed.

### To run locally:
```bash
pip install '../[dashboard]'
streamlit run app.py
```
### Docker:
Use docker-compose in `miade/`:
```bash
cd ..
docker-compose up -d
```
Or standalone:
```bash
cd ..
docker build -f streamlit_app/Dockerfile -t streamlit:miade-dashboard .     
docker run -d --name miade-dashboard -p 8501:8501 streamlit:miade-dashboard
```