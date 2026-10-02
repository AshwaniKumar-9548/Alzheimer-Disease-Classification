# Dataset

The original Alzheimer's MRI image dataset is not stored in this repository.

## Why?

The dataset is kept outside GitHub because the image collection may be large and may have its own licensing, access, privacy, or redistribution conditions.

Obtain the dataset from its authorized/original source and place it locally in this structure:

```text
Dataset/
├── Training Set/
│   ├── <Class 1>/
│   ├── <Class 2>/
│   ├── <Class 3>/
│   └── <Class 4>/
│
└── Testing Set/
    ├── <Class 1>/
    ├── <Class 2>/
    ├── <Class 3>/
    └── <Class 4>/
```

The class names and class ordering must be consistent between Training Set and Testing Set.

The training pipelines automatically read the dataset from:

```text
Dataset/Training Set
Dataset/Testing Set
```
