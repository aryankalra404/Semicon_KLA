FROM nvcr.io/nvidia/pytorch:26.07-py3

WORKDIR /workspace/project
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
