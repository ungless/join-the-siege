import spacy
import pytesseract
import re
import pathlib

from PIL import Image
from werkzeug.datastructures import FileStorage
from pypdf import PdfReader


def get_plaintext_img(file: FileStorage) -> str:
    image = Image.open(file)

    text = pytesseract.image_to_string(image)
    cleaned_text = text.lower()
    return cleaned_text


def get_plaintext_pdf(file: FileStorage) -> str:
    reader = PdfReader(file)
    pages = reader.pages
    output = []
    for page in pages:
        page_text = page.extract_text()
        cleaned_text = re.sub(r'\b([a-zA-Z]) (?=[a-zA-Z]\b)', r'\1', page_text)
        cleaned_text = cleaned_text.lower()
        output.append(cleaned_text)
    return " ".join(output)


def classify_file(file: FileStorage):
    filename = file.filename.lower()
    extension = filename.split(".")[-1]
    if extension == "pdf":
        plaintext = get_plaintext_pdf(file)
    elif extension == "jpg" or extension == "png":
        plaintext = get_plaintext_img(file)
    print(plaintext)
    nlp = spacy.load(pathlib.Path(__file__).parent.resolve() / "output/")

    doc = nlp(plaintext)

    predicted_label = max(doc.cats, key=doc.cats.get)
    return predicted_label
