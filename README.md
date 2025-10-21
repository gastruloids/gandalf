# RingRodeo: domain-specific segmentation of Rod/Ring Structures in Stem Cells

NOTE: If you are looking to run the model, a dockerized version of the app is available [here](https://github.com/stmball/ringrodeo-docker). This is easier to setup, but is a working repository based on feedback. This repository is a snapshot linked to the official implementation of [RingRodeo](https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-025-02226-7).

We include the code for training the model, as well as an accompanying webapp for easy interaction with models for feedback.

## Requirements

### Webapp

For using the webapp, you can install and run the api and frontend servers with the following commands:

```bash
cd webapp/api && poetry install && poetry run python3 app.py
cd webapp/frontend && npm install && npm start
```

### Model development

To run model development for manual inference and training, you can install the requirements via the command:

```bash
cd model_dev && poetry install
```

```

```
