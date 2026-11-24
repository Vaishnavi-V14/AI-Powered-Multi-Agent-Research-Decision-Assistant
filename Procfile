web: streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
api: uvicorn app.api:app --host 0.0.0.0 --port $PORT
