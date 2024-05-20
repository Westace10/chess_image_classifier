FROM python:3.12.3
 
WORKDIR /code
 
COPY ./requirements.txt /code/requirements.txt
 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
 
COPY ./app /code/app

COPY ./strapp /code/strapp

EXPOSE 80
EXPOSE 8501

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 80 & streamlit run /code/strapp/st_app.py --server.port=8501 --server.address=0.0.0.0"]