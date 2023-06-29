FROM python:3.10.9
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "modelling_src/pose_estimation.py"]