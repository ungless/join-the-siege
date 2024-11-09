# Running the app

## Running Training
```sh
docker-compose up train --build
```

## Running the API
```sh
docker-compose up api --build
```

The API accesses a shared docker volume containing the model.

## Why Docker
Containerising our environment here lets the API and model be productionised more quickly. The existing containers can be orchestrated in the production environment without worrying about any differences with the development environment.

## Other changes
I separated `requirements.txt` between the two apps. This lets us create a clean separation between training and the public API environments.

# Model

I used the SpaCy TextCat classifier pipeline with a basic web-based English language model. The TextCatEnsemble model uses a Tok2Vec layer with attention. 

## Synthetic datasets

```sh
systemctl start ollama
ollama pull llama3
python generate.py
```

I used `ollama` to run Llama3 locally to generate synthetic data for the task.

# API

There are two kinds of document ingested: PDFs and images. Text is extracted from the PDFs, it is lowered, and then whitespace is removed for single char words (as is a common result in PDF to text implementations).

Images are represented using Pillow and then passed to a locally-running `tesseract-ocr` service for OCR.

These are then passed to the model in the shared volume and a category is predicted.

# What I would do with more time

I couldn't achieve perfect classification on two of the three drivers licenses. I think this was because the diversity of the outputs here wasn't captured in the synthetically-generated data. With more time I would amend the prompt generating the synthetic data and retrain.

I would also integrate the synthetic data generator with documents seen passing through the service. This way, we could train on documents the system has observed and gather more data.
