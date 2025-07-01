FROM continuumio/miniconda3:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONDA_DEFAULT_ENV=env \
    PATH=/opt/conda/envs/env/bin:$PATH \
    CONDA_PREFIX=/opt/conda/envs/env

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


RUN conda update -n base -c defaults conda && \
    conda install -n base conda-libmamba-solver && \
    conda config --set solver libmamba

RUN conda create -n env python=3.11 -y

RUN echo "conda activate env" >> ~/.bashrc && \
    echo "conda activate env" >> /etc/bash.bashrc

RUN conda init bash && \
    echo "conda activate env" >> ~/.bashrc

SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]

COPY requirements.txt ./
RUN conda run -n env pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
    

USER appuser
RUN conda init bash && \
    echo "conda activate env" >> ~/.bashrc

RUN /bin/bash -c "source ~/.bashrc && which python && python --version"

EXPOSE 8000-8600


# # Set the default command
# CMD ["conda", "run", "--no-capture-output", "-n", "env", "python", "app.py"]