# Cry Baby

Cry Baby provides a probability that your baby is crying by continuously recording audio, chunking it into 4-second clips, and feeding them into a Convolutional Neural Network (CNN).

## Installation

Create a virtual environment and install the dependencies with Poetry using the command poetry install.

Depending on your hardware architecture, Poetry should automatically install the correct version of TensorFlow or TensorFlow Lite. This has been tested on a 2022 M1 MacBook Pro and an Intel NUC running Ubuntu 20.04.

## Usage

You will need a [hugging face account](https://huggingface.co/welcome) and an API token. Once you have the token copy `example.env` to `.env` and add your token there.

A Makefile is provided for running Cry Baby.

```bash
make run
```

Every 4 seconds, Cry Baby will print the probability of a baby crying in each audio clip it records.
