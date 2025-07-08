# Start from a Conda base image which is more suitable for scientific packages
FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Copy the environment file
COPY environment.yml .

# Create the Conda environment from the yml file
# This command also installs all dependencies
RUN conda env create -f environment.yml

# Copy the rest of your project code
COPY . .

# Activate the conda environment and set it as the default shell
SHELL ["conda", "run", "-n", "tinnitus-projekt", "/bin/bash", "-c"]

# The command to run when the container starts
CMD ["python", "train.py"]