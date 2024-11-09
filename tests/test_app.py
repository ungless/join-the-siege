from io import BytesIO

import pytest
from src.app import app, allowed_file


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("file.pdf", True),
        ("file.png", True),
        ("file.jpg", True),
        ("file.txt", False),
        ("file", False),
    ],
)
def test_allowed_file(filename, expected):
    assert allowed_file(filename) == expected


def test_no_file_in_request(client):
    response = client.post("/classify_file")
    assert response.status_code == 400


def test_no_selected_file(client):
    data = {"file": (BytesIO(b""), "")}  # Empty filename
    response = client.post(
        "/classify_file", data=data, content_type="multipart/form-data"
    )
    assert response.status_code == 400


def test_success(client, mocker):
    mocker.patch("src.app.classify_file", return_value="test_class")

    data = {"file": (BytesIO(b"dummy content"), "file.pdf")}
    response = client.post(
        "/classify_file", data=data, content_type="multipart/form-data"
    )
    assert response.status_code == 200
    assert response.get_json() == {"file_class": "test_class"}


def test_file_classification(client):
    files = {
        "invoice": [
            "files/invoice_1.pdf",
            "files/invoice_2.pdf",
            "files/invoice_3.pdf",
        ],
        "bank_statement": [
            "files/bank_statement_1.pdf",
            "files/bank_statement_2.pdf",
            "files/bank_statement_3.pdf",
        ],
        "drivers_licence": [
            "files/drivers_license_1.jpg",
            "files/drivers_licence_2.jpg",
            "files/drivers_license_3.jpg",
        ],
    }
    for category, filenames in files.items():
        for filename in filenames:
            print(filename)
            file = open(filename, "rb")
            data = {"file": (BytesIO(file.read()), filename)}
            response = client.post(
                "/classify_file", data=data, content_type="multipart/form-data"
            )
            assert response.status_code == 200
            assert response.get_json() == {"file_class": category}
