# Preprocessing runtime

[Configuration](benchmark_config.json) · [120 measurements](preprocessing_benchmark_raw.csv) · [Summary by size](preprocessing_benchmark_summary.csv)

Ten assets were measured at each of four input sizes, with three consecutive local reads. The timings cover file read, decode, grayscale conversion and resizing to the working grid. Operating-system cache state was uncontrolled. Model loading, feature extraction and prediction were timed separately in the study.
