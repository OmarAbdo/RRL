# Use a specific version of the rocker/r-ver image for reproducibility
FROM rocker/r-ver:4.2.0

# Install system dependencies required for TensorFlow and other packages
# RUN apt-get update && apt-get install -y \
#     python3-pip \
#     libssl-dev \
#     libcurl4-openssl-dev \
#     && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y python3-pip libssl-dev libcurl4-openssl-dev && pip3 install tensorflow==2.12.0

# Install the R packages required for the script
RUN R -e "install.packages(c('R6', 'ggplot2', 'progress', 'tensorflow', 'keras'))"

# The tensorflow R package will automatically use the python tensorflow already installed


# Set up the working directory
WORKDIR /app

# Copy the R script and the data file into the container
COPY test.r .
COPY data_task4.csv .

# Command to run the R script
CMD ["Rscript", "test.r"]
