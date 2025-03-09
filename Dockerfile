FROM continuumio/miniconda3

# install git
RUN apt-get update && apt-get install -y \
    git

# create a directory for the project
RUN mkdir -p sx263

# copy the project into the container
COPY . /sx263
WORKDIR /sx263

# install the conda envrionment from environment.yml
RUN conda env create -f environment_docker.yml

# can't activate the conda environment in the dockerfile
# activate the pathomic_fusion_hpc conda environment in the bash shell
RUN echo "source activate pathomic_fusion_hpc && conda activate pathomic_fusion_hpc" > ~/.bashrc

# set the default command to bash shell when container is run
SHELL ["/bin/bash", "--login", "-c"]

# set the port to 8888 to connect to Jupyter
EXPOSE 8888