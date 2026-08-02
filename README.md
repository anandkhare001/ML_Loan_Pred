# Loan Prediction API

A FastAPI-based machine learning application that predicts whether a loan application is likely to be approved or rejected using a trained Random Forest model.

## Overview

This project loads a pre-trained model from `RF_Loan_model.pkl` and exposes it through a REST API. The service accepts applicant information and returns a simple approval decision.

The app is designed for local development, Docker-based deployment, and basic Kubernetes deployment examples.

## Tech Stack

- Python 3.11
- FastAPI
- Uvicorn
- Pydantic
- scikit-learn
- NumPy
- Pandas
- Pytest

## Project Structure

```text
.
├── app_fast_local.py       # FastAPI application and prediction logic
├── app_k8.yaml             # Kubernetes deployment/service example
├── config.yaml             # Deployment config placeholder
├── Dockerfile              # Container setup for the app
├── requirements.txt        # Python dependencies
├── RF_Loan_model.pkl       # Trained Random Forest model
├── test_models.py          # Basic model validation test
└── README.md               # Project documentation
```

## Features

- REST API endpoint for loan prediction
- Validation of input payload using Pydantic models
- Model inference using a pre-trained machine learning pipeline
- Container-ready Docker configuration
- Sample Kubernetes manifest for deployment

## Local Setup

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate the environment:

On Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
python app_fast_local.py
```

The API will run on:

```text
http://127.0.0.1:5000
```

## API Endpoints

### GET /

Returns a welcome message.

Example response:

```json
{
  "message": "Welcome to Loan Prediction App"
}
```

### POST /prediction

Accepts loan application details and returns whether the loan is approved or rejected.

Request body example:

```json
{
  "Gender": 1,
  "Married": 1,
  "Dependents": 1,
  "Education": 1,
  "Self_Employed": 0,
  "LoanAmount": 120,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": 0,
  "TotalIncome": 3000
}
```

Example response:

```json
{
  "Status of Loan Application": "Approved"
}
```

The app maps the prediction result as follows:

- `0` -> `Rejected`
- `1` -> `Approved`

## Model Input Fields

The prediction endpoint expects the following fields:

- `Gender`: numeric value for gender
- `Married`: marital status
- `Dependents`: number of dependents
- `Education`: education level
- `Self_Employed`: self-employment status
- `LoanAmount`: requested loan amount
- `Loan_Amount_Term`: loan term in months
- `Credit_History`: credit history indicator
- `Property_Area`: property area category
- `TotalIncome`: total applicant income

## Docker Usage

Build the image:

```bash
docker build -t loan-pred .
```

Run the container:

```bash
docker run -p 5000:5000 loan-pred
```

This will expose the app on the same local port used by the FastAPI service.

## Kubernetes

A sample deployment is included in `app_k8.yaml`.

To apply it:

```bash
kubectl apply -f app_k8.yaml
```

This manifest provides a basic Deployment and Service for running the app in a Kubernetes cluster. Note that the app itself listens on port `5000`, so port settings should be reviewed carefully against your cluster environment before production use.

## Testing

Run the basic validation test:

```bash
pytest test_models.py
```

The test loads the trained model and checks a sample loan application prediction.

## Notes

- The application uses a pre-trained model file included in the repository.
- The project is a lightweight example for loan approval classification and is intended for learning, local deployment, and experimentation.
- If you want to extend it, you can replace the model file and update the request schema as needed.

## License

This project does not currently declare a formal license file. Please confirm repository ownership or licensing requirements before using it in a production or public environment.
