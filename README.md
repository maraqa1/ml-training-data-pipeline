# **Topic Modeling and Clustering Pipeline**

This repository contains an end-to-end pipeline for topic modeling and clustering using enriched text features, optimized dimensionality reduction, and iterative clustering refinement. The pipeline processes large text datasets, extracts meaningful topics, and assigns concise labels for each cluster using advanced NLP techniques and APIs.

---

## **Table of Contents**

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Pipeline Workflow](#pipeline-workflow)
6. [Detailed Steps](#detailed-steps)
7. [Outputs](#outputs)
8. [Optimization Techniques](#optimization-techniques)
9. [Reproducibility Guidance](#reproducibility-guidance)
10. [License](#license)

---

## **Introduction**

This project focuses on building a robust pipeline for processing and clustering textual data. It uses:

- **Text cleaning** with custom and library-based functions.
- **NLP models** for extracting enriched features like domains, sectors, and hierarchical topics.
- **Clustering algorithms** such as K-means, optimized with dimensionality reduction techniques like TF-IDF.
- **Refinement techniques** using LDA for sub-clustering low-coherence clusters.
- **Parallel processing** for efficient handling of large datasets.

The primary goal is to create meaningful clusters and label them for interpretability and downstream analysis.

---

## **Features**

- **Text Preprocessing**: Comprehensive text cleaning and tokenization.
- **Feature Extraction**: Use of OpenAI API to extract hierarchical topics and sectors.
- **Efficient Clustering**: Dimensionality reduction with TF-IDF and optimized K-means.
- **Cluster Refinement**: Iterative LDA-based sub-clustering for granular topics.
- **Top Terms Extraction**: Identification of key terms per cluster for interpretability.
- **Cluster Labeling**: Automated labeling using ChatGPT and user-provided labels.
- **Parallel Processing**: Scalable processing using `furrr` and `future` libraries.
- **Customizable**: Supports different datasets and customizable preprocessing options.

---

## **Installation**

### Prerequisites

Ensure the following are installed on your system:
- R (version 4.0 or later)
- Required R libraries (listed below)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/username/repo-name.git
   cd repo-name
   ```

2. Install required R packages:
   ```R
   install.packages(c("tidyverse", "tm", "textclean", "tidytext", "quanteda", "cluster", "udpipe", "furrr", "progressr", "textmineR"))
   ```

3. Download and configure the Udpipe English model:
   ```R
   ud_model <- udpipe_download_model(language = "english")
   ud_model <- udpipe_load_model(file = ud_model$file_model)
   ```

---

## **Usage**

### Preprocessing and Feature Extraction

1. Prepare your dataset as a CSV file with at least the following column:
   - `PublicDescription`: A column containing textual descriptions.

2. Use the provided functions to clean and preprocess the text, extract features, and create a TF-IDF matrix.

3. Apply clustering algorithms to group similar documents.

### Example Workflow

```R
process_large_file(
  input_file = "path/to/input.csv",
  output_dir = "path/to/output/",
  api_key = "your_openai_api_key",
  sample_size = 1000,  # Optional
  chunk_size = 100     # Adjust as needed
)
```

### Generating Cluster Labels

Extract cluster terms and generate concise labels using ChatGPT:

```R
top_terms_df <- extract_cluster_terms(tfidf_reduced, new_kmeans_model, top_n = 30)
```

Refine clusters using LDA for sub-clustering:

```R
new_clusters <- split_clusters_with_lda(tfidf_reduced, kmeans_model_updated, k = 5, coherence_threshold = 0.08)
```

Save the results:

```R
saveRDS(new_kmeans_model, file = "output/models/kmeans_model.rds")
saveRDS(tfidf_reduced, file = "output/models/tfidf_reduced.rds")
```

---

## **Pipeline Workflow**

### Overview

1. **Data Loading**:
   - Read the dataset from a CSV file.
   - Combine relevant columns into a single text field.

2. **Text Preprocessing**:
   - Clean text to remove unnecessary elements.
   - Extract linguistic features (nouns, verbs) using Udpipe.

3. **Feature Engineering**:
   - Apply TF-IDF to create a document-feature matrix.
   - Reduce the matrix size for clustering.

4. **Clustering**:
   - Perform K-means clustering with an optimized number of clusters (`k`).
   - Extract top terms for each cluster for interpretability.

5. **Cluster Refinement**:
   - Use LDA to refine clusters with low coherence scores.

6. **Cluster Labeling**:
   - Generate concise and meaningful labels for each cluster.

7. **Save Outputs**:
   - Save the final clustering model and TF-IDF matrix for reuse.

---

## **Detailed Steps**

### Step 1: Data Preparation

- Ensure the dataset is clean and structured.
- Example structure:
  | doc_id | PublicDescription         |
  |--------|---------------------------|
  | 1      | "Analyzing renewable energy..." |
  | 2      | "Exploring AI in healthcare..." |

### Step 2: Text Preprocessing

- Clean the text to standardize formatting.
- Remove stopwords, punctuation, and numbers.
- Extract meaningful words using the Udpipe model.

### Step 3: Feature Engineering

- Generate a document-feature matrix using TF-IDF.
- Tokenize text into unigrams, bigrams, and trigrams.
- Filter the top terms to reduce matrix size.

### Step 4: Clustering

- Apply K-means clustering to group documents.
- Optimize the number of clusters using Silhouette scores.

### Step 5: Cluster Refinement

- Use LDA for sub-clustering large or low-coherence clusters.
- Reassign documents to refined clusters.

### Step 6: Labeling and Saving Results

- Extract top terms for each cluster.
- Use ChatGPT or manual methods to assign meaningful labels.
- Save the final clustering model and associated data.

---

## **Outputs**

### Files

1. **Clustered Data**:
   - File: `output/consolidated_results.csv`
   - Description: Contains enriched features and cluster assignments.

2. **K-means Model**:
   - File: `output/models/kmeans_model.rds`
   - Description: Includes updated cluster centers and labels.

3. **TF-IDF Matrix**:
   - File: `output/models/tfidf_reduced.rds`
   - Description: Reduced matrix for clustering.

### Logs

- Debugging logs are generated for each step of the pipeline.

---

## **Optimization Techniques**

- **Parallel Processing**:
  - Use `furrr` and `future` for parallel execution of heavy tasks.

- **Dimensionality Reduction**:
  - Retain only top TF-IDF terms to improve clustering performance.

- **Cluster Refinement**:
  - Apply LDA for sub-clustering and improving topic granularity.

---

## **Reproducibility Guidance**

1. **Dependencies**:
   - List all required libraries in your R environment.

2. **Environment Setup**:
   - Use consistent seeds for reproducibility.
   - Configure the environment for parallel processing.

3. **Version Control**:
   - Track all changes using Git.

4. **Testing**:
   - Validate the pipeline on smaller datasets before scaling.

---

## **License**

This project is licensed under the MIT License. See the LICENSE file for details.
