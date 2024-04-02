# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.9.6
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install gcc and other dependencies required for building certain Python packages
# Install gcc, make, wget and other dependencies required for building certain Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    libc-dev \
    make \
    wget \
    && rm -rf /var/lib/apt/lists/*



# Download and compile SQLite from source
RUN wget https://www.sqlite.org/2024/sqlite-autoconf-3450200.tar.gz \
    && tar xvfz sqlite-autoconf-3450200.tar.gz \
    && cd sqlite-autoconf-3450200 \
    && ./configure --prefix=/usr --disable-static --enable-fts5 \
    && make \
    && make install \
    && cd .. \
    && rm -rf sqlite-autoconf-3450200 sqlite-autoconf-3450200.tar.gz

# Confirm SQLite version
RUN sqlite3 --version

# Updated to install requirements using pip directly without bind mount,
# as the requirements.txt should be copied into the image in the next steps.
# This change assumes requirements.txt is part of the context being sent to Docker daemon.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Switch to the non-privileged user to run the application.
USER appuser

# Copy the source code into the container.
COPY . .

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
CMD streamlit run app.py --server.port $PORT

