# Amharic Sign Language Transcription

This project aims to transcribe Amharic Sign Language into text/speech using machine learning and computer vision.

## Project Structure

```text
.
├── data/
│   ├── external/       # Data from third party sources.
│   ├── processed/      # The final, canonical data sets for modeling.
│   └── raw/            # The original, immutable data dump.
├── docs/               # Project documentation.
├── models/             # Trained and serialized models, model predictions, or model summaries.
├── notebooks/          # Jupyter notebooks.
├── src/                # Source code for use in this project.
├── tests/              # Unit and integration tests.
├── .gitignore          # Standard Python .gitignore.
├── README.md           # The top-level README for developers using this project.
└── requirements.txt    # The requirements file for reproducing the analysis environment.
```

## Getting Started

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Jupyter notebooks in `notebooks/` to explore the data or train models.

## Dependencies

-   mediapipe
-   opencv-python
-   psutil
-   jupyter
-   notebook
