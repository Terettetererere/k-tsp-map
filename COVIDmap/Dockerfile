FROM python:3.8

WORKDIR /work

COPY requirements.txt /work

RUN pip install -U pip \
  && pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py"]