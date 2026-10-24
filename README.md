# Smart Shelf AI

Smart Shelf AI is a computer vision project for detecting and counting
nine product classes in shelf images. It uses a trained YOLOv8n model
and a Streamlit interface for image-based inference.

## Features

- Upload a shelf image through the Streamlit application.
- Detect and count supported product classes.
- Display model predictions using the inference module.
- Store evaluation summaries in CSV format.

## Project Structure

```text
smart-shelf-ai/
├── app/
│   ├── app.py
│   ├── inference.py
│   ├── requirements.txt
│   └── yolov8n_9class_v3_best.pt
├── src/
├── notebooks/
├── results/
│   ├── overall_metrics.csv
│   ├── per_class_metrics.csv
│   ├── per_image_metrics.csv
│   ├── error_diagnostics.csv
│   └── model_comparison.csv
├── research/
├── .gitignore
├── README.md
└── requirements.txt
```

## Run the Application

1. Install Python and create a virtual environment.
2. Install the application dependencies:

   ```bash
   pip install -r app/requirements.txt
   ```

3. Start the application from the app directory:

   ```bash
   cd app
   streamlit run app.py
   ```

The trained checkpoint must remain beside `inference.py`.

## Evaluation Results

The supplied evaluation summary reports the following YOLOv8n results:

| Metric | Value |
|---|---:|
| True positives | 21 |
| False positives | 5 |
| False negatives | 5 |
| Precision | 0.8077 |
| Recall | 0.8077 |
| Micro F1 | 0.8077 |

These figures describe the supplied evaluation file and should be
interpreted in the context of its evaluation sample size. They do not
establish performance on all retail environments.

The repository also contains per-class metrics, per-image metrics,
error diagnostics, and a model-comparison CSV.

## Limitations

- Detection is limited to the nine classes supported by the checkpoint.
- Performance may vary with shelf layout, lighting, occlusion, and image quality.
- Evaluation results depend on the size and quality of the labelled test sample.

## Reproducibility

The deployment code and model checkpoint are included in the `app/`
directory. Refer to `app/README.md` for the original deployment instructions.

## License

No license has been selected yet. All rights remain with their respective
owners until a license is added.
