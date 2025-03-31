
# 📘 NLP Clustering & Topic Modeling Pipeline

## 🎯 Objective

This project provides a robust NLP pipeline for clustering and labeling
large volumes of text data. It is built for scalability,
interpretability, and seamless integration with the `SectorInsightRv2`
platform.

The pipeline enables: - Extraction of structured topics and sectors
using OpenAI GPT. - Document grouping via TF-IDF + K-means clustering. -
Evaluation and refinement of clusters based on semantic coherence. -
Label generation using GPT for thematic clarity. - Export of
ready-to-use models for downstream prediction and analytics.

------------------------------------------------------------------------

## 🧠 Design Philosophy

This system adopts a hybrid NLP strategy combining deterministic
clustering, probabilistic topic modeling, and LLM-powered semantics.

| Layer                   | Description                                                   |
|--------------------------|----------------------------------------------|
| **Feature Enrichment**  | Topic & sector extraction using GPT                           |
| **Linguistic Cleaning** | POS filtering (NOUNs + VERBs) using UDPipe                    |
| **Vectorization**       | TF-IDF weighted n-grams (1–3)                                 |
| **Initial Clustering**  | Fast, scalable K-means clustering                             |
| **Refinement**          | Adaptive LDA with coherence thresholds to split weak clusters |
| **Labeling**            | GPT-generated labels based on cluster term distributions      |
| **Integration**         | Final models exported for `SectorInsightRv2` deployment       |

------------------------------------------------------------------------

## 🧱 Pipeline Components

| File                                                    | Purpose                                                               |
|---------------------------------|---------------------------------------|
| `01_extract_features.R`                                 | Enriches data with GPT-based topics/sectors; cleans text using UDPipe |
| `02_build_models.R`                                     | Standard clustering pipeline with TF-IDF + K-means + GPT labeling     |
| `02_build_models_with_auto_lables_optimised_clusters.R` | Optimized version with LDA-based splitting and merging                |
| `04_cluster_refiner.R`                                  | Post-hoc refinement and evaluation based on topic coherence           |

## 📦 Output & Integration with SectorInsightRv2

The final artifacts from this pipeline are saved under the `output/`
directory and are compatible with the `SectorInsightRv2` package.

### **Output Files and Locations**

| **Directory**    | **Description**                                       |
|------------------|-------------------------------------------------------|
| `output/kmeans/` | Optimized `kmeans` models with cluster IDs and labels |
| `output/dfm/`    | Sparse TF-IDF matrices (`dgCMatrix`) for predictions  |

> 🗂 Note: Filenames are NOT dynamically generated. you have to manually
> set the timestamps or version numbers).

## 🔧 Requirements

``` r
# Minimum R Version
R >= 4.1

# Required Packages
install.packages(c(
  "tidyverse", 
  "tm", 
  "quanteda", 
  "textclean", 
  "textmineR", 
  "furrr", 
  "httr", 
  "jsonlite", 
  "progressr"
))

# Language Model
# Download and load the English UDPipe model (run once per environment)
library(udpipe)
ud_model <- udpipe_download_model(language = "english")
```

## 🔭 Next Sections

The pipeline is organized across multiple scripts, each focusing on a
specific stage of the workflow. Below are the next sections in the
documentation:

### **Part 2 — Feature Enrichment Pipeline**

📂 `01_extract_features.R`\
Uses OpenAI GPT to extract structured topic and sector information.
Cleans and filters text using UDPipe (NOUNs + VERBs). Outputs are saved
and passed to the modeling stage.

------------------------------------------------------------------------

### **Part 3 — Modeling & Optimization**

📂 `02_build_models.R`\
The base modeling script: builds TF-IDF features, applies K-means
clustering, and labels clusters LDA modelling) and uses GPT-generated
cluster lables.

📂 `02_build_models_with_auto_lables_optimised_clusters.R`\
Extended version that introduces: - Adaptive LDA splitting for
low-coherence clusters - Centroid similarity-based cluster merging -
Automated GPT labeling after refinement

### **📦 Part 4 — Cluster Refinement & Evaluation**

📂 `04_cluster_refiner.R`\\ Standalone utility functions for: -
Re-assessing coherence of existing clusters - Further breaking down
ambiguous or low-quality groupings - Generating final coherence
summaries for validation and QA

## 📦 Part 5 — Integration with SectorInsightRv2

------------------------------------------------------------------------

## 📦 Part 2 — Feature Enrichment Pipeline

### 📂 `01_extract_features.R`

This script is responsible for preprocessing raw document descriptions
and extracting structured features using OpenAI GPT and UDPipe.

### **Purpose**

To convert raw `PublicDescription` fields into semantically enriched
features for clustering. This is the interface between unstructured
input and structured modeling.

------------------------------------------------------------------------

### **Key Steps**

#### ✅ 1. Text Cleaning (`clean_text`)

-   Converts text to lowercase
-   Removes punctuation, numbers, stopwords
-   Removes domain-agnostic noise words (e.g., “project”, “team”)

> **Output:** `cleaned_text` — cleaned version of the input text

------------------------------------------------------------------------

#### ✅ 2. OpenAI-Powered Feature Extraction

Uses OpenAI GPT (e.g., `gpt-3.5-turbo-instruct`) to extract: -
`Domain` - `Level_1`, `Level_2` - `Primary_Sector`, `Secondary_Sector`

> These are combined into a single column: `combined_features`

------------------------------------------------------------------------

#### ✅ 3. Linguistic Enrichment (`cleaned_text_1`)

-   UDPipe extracts only **NOUNs and VERBs** from `combined_features`
-   Reduces dimensionality and noise by excluding adjectives, modifiers,
    etc.
-   Parallelized with `{furrr}` and `{progressr}`

> **Output:** `cleaned_text_1` — linguistically enriched version used to
> build the DFM

------------------------------------------------------------------------

### **Why `cleaned_text_1` Matters**

-   More focused than raw text
-   Emphasizes semantically rich tokens
-   Produces cleaner clusters and better labels

------------------------------------------------------------------------

### **Output**

-   `output/features.csv`: contains `Domain`, `Level_1`, `Level_2`,
    `Primary_Sector`, `Secondary_Sector`, `combined_features`,
    `cleaned_text_1`
    ------------------------------------------------------------------------

## 📦 Part 3 — Modeling & Optimization

------------------------------------------------------------------------

### 📂 `02_build_models.R`

The base modeling script. It builds TF-IDF features, applies K-means
clustering, and introduces **adaptive LDA-based splitting** for
low-coherence clusters.

------------------------------------------------------------------------

### **Key Steps**

#### ✅ TF-IDF Feature Matrix

-   Tokenizes `cleaned_text_1` into n-grams (1–3)
-   Weights tokens with TF-IDF
-   Reduces dimensionality using top `n` tokens (e.g., 1000–2000)

> **Output:** `tfidf_reduced` — used for clustering

------------------------------------------------------------------------

#### ✅ K-means Clustering

-   Applies **K-means** to the reduced TF-IDF matrix (`tfidf_reduced`)
    to identify thematic groupings. The choice was 15 clusters. This was
    a manual iterative process.
-   Each document is assigned a **hard cluster label** (i.e., belongs to
    one and only one cluster).
-   **Centroids** are calculated for each cluster — these are numeric
    vectors representing the "center" of each cluster in feature space.
-   The **top terms** per cluster are extracted from centroid weights,
    revealing the most important features driving each grouping.

> **Outputs:** - `kmeans_model$cluster`: Integer vector of cluster
> assignments (1 per document) - `kmeans_model$centers`: Matrix of
> cluster centroids (used to extract keywords) - `top_terms_df`:
> DataFrame mapping each cluster to its top n terms (e.g., top 30
> keywords) ---

#### ✅ Adaptive LDA-Based Sub-Clustering

-   Evaluates each K-means cluster's **semantic coherence** using
    `textmineR::CalcProbCoherence()`.
-   If a cluster’s average coherence is **below a specified threshold**
    (e.g., 0.05), it is split using **adaptive LDA**:
    -   Tries multiple topic counts (k = 2:6)
    -   Chooses the best `k` with the highest average coherence
-   Documents are reassigned into **refined sub-clusters** based on LDA
    topic distributions.

> **Outputs:** - `new_clusters`: Updated vector of cluster assignments
> (includes original + subclusters) - `coherence_scores`: Numeric score
> per cluster measuring interpretability - `refined_model`: Updated
> K-means-like object reflecting the LDA-enhanced structure

------------------------------------------------------------------------

#### ✅ GPT-Based Labeling

-   From each (sub)cluster, **top TF-IDF terms** are extracted to
    represent its thematic content.
-   These term lists are sent to the **OpenAI ChatGPT API** with prompts
    requesting short, descriptive **cluster labels**.
-   GPT returns **natural-language summaries** (e.g., "Climate
    Innovation in Agriculture") for each cluster.

> **Outputs:** - `cluster_labels_Spacy`: Named vector of cluster labels
> (e.g., `Cluster_1 = "Digital Manufacturing"`) - `kmeans_model$labels`:
> Label applied to each document based on its cluster -
> `chatgpt_prompt`: Optional exported file with full prompt content for
> reproducibility

------------------------------------------------------------------------

### ✅ Why Use K-means First?

-   **Scalable for Large Text Data:** Efficient on high-dimensional
    TF-IDF matrices with large corpora.
-   **Hard Clustering:** Assigns each document to a single cluster,
    simplifying coherence evaluation and downstream labeling.
-   **Centroid Interpretability:** Produces clear centroids for each
    cluster, enabling keyword extraction and cluster merging.
-   **Ideal First-Pass Grouping:** Quickly separates documents into
    broad thematic groups, which can later be refined with LDA.

------------------------------------------------------------------------

### 🔍 Why Not LDA First?

-   **Slower and Resource-Intensive:** Computationally expensive on
    large vocabularies.
-   **Sensitive to Hyperparameters:** Poor initial topic counts can
    reduce coherence.
-   **Soft Clustering Ambiguity:** Probabilistic assignments make
    document-level labeling less straightforward.
-   **Better as a Refinement Tool:** Performs best when applied
    selectively to noisy or mixed clusters after initial segmentation.

### 📂 `02_build_models_with_auto_lables_optimised_clusters.R` \|

| This is an enhanced version of `02_build_models.R`, extending the clustering pipeline by adding **automated refinement, merging, and relabeling**. It aims to improve **semantic coherence**, **reduce redundancy**, and **maximize interpretability** using adaptive logic and GPT-based labeling. \|

### 🔧 Key Additions & Improvements

#### ✅ **Adaptive LDA-Based Sub-Clustering**

-   Applies LDA selectively to clusters with low coherence (using
    `CalcProbCoherence()`).
-   Dynamically chooses the best topic count (`k`) from a range (e.g.,
    2–6).
-   Splits incoherent clusters into finer, more semantically aligned
    subclusters.

#### ✅ **Cluster Merging Based on Cosine Similarity**

-   Compares centroids of all clusters using **cosine similarity**.
-   Automatically **merges semantically similar clusters** (above a
    user-defined threshold, e.g., 0.90).
-   Prevents over-fragmentation and ensures a cleaner final structure.

#### ✅ **Re-labeling After Merging**

-   After clusters are split or merged, new top TF-IDF terms are
    extracted.
-   GPT (e.g., `gpt-4`) is used again to generate **refreshed,
    meaningful labels** for each final cluster.
-   Ensures updated clusters reflect accurate and concise topic
    summaries.

------------------------------------------------------------------------

### 🔄 **Pipeline Flow in This Script**

it can be used instead of the
`02_build_models.R as it has all the key processing elements as below:`

1.  **K-means Clustering**\
    ➝ Efficient initial grouping using reduced TF-IDF features.

2.  **LDA-Based Sub-Clustering**\
    ➝ Splits noisy clusters adaptively based on coherence evaluation.

3.  **Centroid-Based Cluster Merging**\
    ➝ Merges redundant clusters using cosine similarity.

4.  **GPT-Based Re-labeling**\
    ➝ Final semantic labels generated using top cluster terms.

------------------------------------------------------------------------

### 💾 Output Files

-   `final_kmeans_model_optimized.rds`\
    ➝ Final clustering model with coherent, labeled groupings.

-   `final_tfidf_reduced_matrix.rds`\
    ➝ Reduced TF-IDF matrix used throughout the pipeline.

> These files are intended for integration into the `SectorInsightRv2`
> package.

------------------------------------------------------------------------

### 🧠 Integrated Design Philosophy

-   **K-means:** Provides scalable, hard clustering to initialize the
    pipeline.
-   **LDA:** Improves thematic resolution by refining low-quality
    clusters.
-   **Merging:** Removes redundancy by collapsing highly similar
    centroids.
-   **GPT:** Translates statistical clusters into **human-readable
    labels** for interpretability.

> Together, these steps create a pipeline that is robust, interpretable,
> and production-ready for scalable sectoral insight generation.

### **📦 Part 4 — Cluster Refinement & Evaluation**

📂 `04_cluster_refiner.R`\
A standalone script containing utility functions for **analyzing**,
**refining**, and **validating** cluster quality after the main modeling
phase.

------------------------------------------------------------------------

### 🔍 Purpose

This module is designed for **post-hoc analysis** and refinement of
existing clusters generated from earlier stages (`02_build_models.R` or
`02_build_models_with_auto_lables_optimised_clusters.R`). It enables:

-   Reassessment of **semantic coherence**
-   Further **splitting of noisy clusters** via adaptive LDA
-   **Summary reporting** of coherence scores across all clusters

------------------------------------------------------------------------

### ⚙️ Key Capabilities

#### ✅ `summarise_cluster_coherence()`

-   Evaluates **topic coherence** for each cluster using LDA.
-   Uses `textmineR::CalcProbCoherence()` to quantify how semantically
    tight a cluster is.
-   Returns a **summary dataframe** with:
    -   Cluster ID
    -   Document count
    -   Average coherence score
    -   Associated GPT label

> Useful for **QA and validation**, or identifying clusters needing
> refinement.

------------------------------------------------------------------------

#### ✅ `split_clusters_with_lda_adaptive()`

-   Selectively applies LDA to **low-coherence clusters**.
-   Dynamically determines the best topic count (`k`) from a defined
    range (e.g., 2–6).
-   Splits clusters **only when coherence is below a threshold** (e.g.,
    0.05).
-   Returns **new cluster assignments** with improved thematic clarity.

------------------------------------------------------------------------

#### ✅ `update_kmeans_model()`

-   Recalculates cluster centroids and sizes based on updated
    assignments.
-   Ensures that the cluster model remains consistent and correctly
    structured.
-   Also updates `tot.withinss`, `centers`, and `size` attributes for
    downstream use.

------------------------------------------------------------------------

### 📊 Output & Use Cases

-   **Cluster-Level Coherence Report:**\
    Helps diagnose poor clustering or overlapping topics.

-   **Refined Cluster Assignments:**\
    Can be merged back into the final model or used to inform future
    iterations.

-   **Integration:**\
    These utilities can be embedded into the end of
    `02_build_models_with_auto_lables_optimised_clusters.R` or run
    independently during QA review cycles.

> 🧠 All outputs from these steps are saved to `output/kmeans/` and
> `output/dfm/`, and are designed for plug-and-play use within
> `SectorInsightRv2`

## 📦 Part 5 — Integration with SectorInsightRv2

This section addresses how models generated in `02_build_models.R` and
`02_build_models_with_auto_lables_optimised_clusters.R` are integrated
into the **SectorInsightRv2** package.

These scripts produce the core modeling outputs:

 - A refined **K-means model** with document cluster assignments and
GPT-generated labels.

\- A **TF-IDF matrix (DFM)** based on linguistically filtered features
(NOUNs + VERBs via UDPipe). These outputs must be stored in a specific
structure with defined naming conventions to enable seamless use within
the package for classification, filtering, and dashboard visualizations.

------------------------------------------------------------------------

### 📁 Output Naming Convention

To ensure compatibility with the `SectorInsightRv2` internal loaders,
all output model files must follow this naming scheme:

-   **K-means clustering model:**: xx_topic_kmeans.rds

-   **TF-IDF document-feature matrix:** : xx_topic_dfm.rds

Where `xx` is a version or model ID number (e.g., `43_topic_kmeans.rds`,
`43_topic_dfm.rds`).

--

### 📂 Model File Location

These files must be copied to the following directory inside the
package:

-   **K-means clustering model:**:
    SectorInsightRv2/inst/extdata/models/kmeans/xx_topic_kmeans.rds
-   **TF-IDF document-feature matrix:**:
    SectorInsightRv2/inst/extdata/models/dfm/xx_topic_dfm.rds

### Package Rebuild and Export

After copying the model files, rebuild the package using devtools:

``` r

devtools::document()
devtools::install()
```

## **Integration with SectorInsightRv2**

### **Usage of DFM and K-means Models**

1.  **DFM for Topic Analysis:**
    -   The DFM can be loaded into SectorInsightRv2 to analyze term
        distributions across topics.
    -   Enables visualization of term importance and facilitates
        keyword-based filtering.
2.  **K-means Model for Classification:**
    -   Use the K-means model to classify new documents into existing
        clusters.
    -   Leverage centroids to identify the closest topic for a given
        document.

#### **Example of using the models:**

``` r

# Load the models from within the installed package
kmeans_model <- readRDS(system.file("extdata/models/43_topic_kmeans.rds", package = "SectorInsightRv2")) 
dfm <- readRDS(system.file("extdata/models/43_topic_dfm.rds", package = "SectorInsightRv2"))
# Predict a cluster for a new document
text <- "Exploring new AI-driven diagnostics for medical imaging" tokens <- quanteda::tokens(text, ngrams = 1:3)
dfm_new <- quanteda::dfm(tokens) dfm_new <- quanteda::dfm_match(dfm_new, features = colnames(dfm))
cluster <- stats::predict(kmeans_model, newdata = as.matrix(dfm_new)) cat("Predicted Cluster:", cluster)
```

------------------------------------------------------------------------

## **Reproducibility Guidance**

1.  **Version Control:**
    -   Track all changes using Git.
2.  **Environment Setup:**
    -   Ensure required R libraries are installed.
3.  **Random Seeds:**
    -   Set random seeds for clustering to ensure consistent results.
4.  **Testing:**
    -   Validate the pipeline with a small dataset before scaling up.

------------------------------------------------------------------------
