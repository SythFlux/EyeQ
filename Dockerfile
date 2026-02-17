FROM dustynv/pytorch:2.7-r36.4.0

RUN apt-get update && apt-get install -y python3-opencv

RUN pip3 install --no-cache-dir --index-url https://pypi.org/simple --ignore-installed flask

# key line: no deps, so it won't install torch/torchvision/opencv-python
RUN pip3 install --no-cache-dir --index-url https://pypi.org/simple --no-deps ultralytics

WORKDIR /app
