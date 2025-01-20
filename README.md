# **Topic Modeling and Clustering Pipeline Documentation**

## **Objective**
To process text data and cluster documents into meaningful topics using enriched features generated with the OpenAI API. The clustering outputs include a Document-Feature Matrix (DFM) and a refined K-means model, optimized for computational efficiency and coherence. These models will be integral to the **SectorInsightRv2** package for downstream analysis and deployment.

---

## **Pipeline Overview**

This pipeline consists of the following stages:

1. **Data Ingestion**
2. **Text Preprocessing**
3. **Chunking and Size Optimization**
4. **Feature Engineering with OpenAI API**
5. **Building the Document-Feature Matrix (DFM)**
6. **Clustering**
7. **Post-Clustering Enhancements**
8. **Refinement of Clusters**
9. **Label Assignment**
10. **Saving Models for SectorInsightRv2**

---

## **Detailed Stages**

### **1. Data Ingestion**
**Input:** CSV file containing raw text (`PublicDescription`) and metadata.

**Process:**
- Load the CSV file into a dataframe.
- Validate the data (e.g., check for missing or malformed text entries).

**Output:** A dataframe with `PublicDescription` and other fields ready for processing.

---

### **2. Text Preprocessing**
**Purpose:** Normalize text and prepare it for feature extraction by cleaning and simplifying the raw input.

**Steps:**
1. Convert text to lowercase to ensure uniformity.
2. Remove punctuation, stopwords, and numbers to focus on meaningful terms.

**Output:** A dataframe with cleaned text stored in the `cleaned_text` column, ready for subsequent processing.

---

### **3. Chunking Data for API Processing**

**Purpose:** Split the data into smaller, manageable chunks to efficiently process it through the OpenAI API while adhering to API limits and optimizing memory usage.

#### **Steps:**
1. Divide the dataset into chunks of a specified size (e.g., 1000 rows per chunk) to streamline processing.
2. Send each chunk to the OpenAI API for feature extraction, ensuring compliance with API rate limits.
3. Collect and consolidate the results from all chunks into a single dataframe for subsequent processing.

**Advantages:**
- Manages memory efficiently by processing smaller subsets.
- Facilitates parallel processing to speed up API calls.

---

### **4. Feature Engineering with OpenAI API**
**Purpose:** Use OpenAI's API to generate text features. For this purpose, an API key has to be generated. Follow the instructions at [OpenAI API Key Setup](https://platform.openai.com/account/api-keys) to create your API key.

**Steps:**
- Extract hierarchical topics from `PublicDescription`:
  - Domain
  - Level 1
  - Level 2
- Identify sectors:
  - Primary Sector
  - Secondary Sector
- Combine extracted features with `PublicDescription` to form a `combined_features` column.

**Output:** A dataframe with `Domain`, `Level 1`, `Level 2`, `Primary Sector`, `Secondary Sector`, and `combined_features` ready for clustering.

---

### **5. Building the Document-Feature Matrix (DFM)**

**Purpose:** Use the consolidated API results to create a unified Document-Feature Matrix (DFM) for clustering.

#### **Steps:**
1. Use the consolidated results from the API to generate the DFM:
   - Tokenize `combined_features` into unigrams, bigrams, and trigrams.
   - Extract nouns and verbs from `combined_features` using the `udpipe` model. These linguistic features are stored in the `cleaned_text_1` column.
   - Apply TF-IDF weighting to emphasize distinctive terms.
2. Retain the top 1000 terms with the highest TF-IDF scores across the entire dataset:
   - Compute column sums of the DFM to identify frequent terms.
   - Filter terms to create a reduced DFM.

**Advantages:**
- Focuses on the most relevant terms for clustering.
- Reduces computational load while maintaining meaningful features.

---

### **6. Clustering**

#### **Building the K-means Model**
**Purpose:** Cluster documents into topics based on the reduced DFM.

**Process:**
1. Set the number of clusters (`k`) based on evaluation metrics such as:
   - Silhouette score.
   - Elbow method.
2. Train a K-means model using the reduced DFM.
3. Assign documents to clusters and calculate centroids.

**Output:**
- Cluster assignments for each document.
- Cluster centroids representing the central term distributions.

---

### **7. Post-Clustering Enhancements**

#### **Cluster Analysis**
1. Extract top terms for each cluster from the centroids:
   - Rank terms by their importance within the cluster.
   - Identify shared terms to assess overlap across clusters.

2. Evaluate cluster coherence:
   - Analyze intra-cluster similarity.
   - Use external validation (e.g., domain-specific knowledge).

#### **LDA-based Sub-clustering**
1. Apply Latent Dirichlet Allocation (LDA) to refine large or incoherent clusters:
   - Identify subtopics within clusters.
   - Split clusters where average topic coherence is below a threshold.
2. Assign documents to sub-clusters and update centroids.

#### **Recalculate Cluster Centers**
1. Update centroids to reflect the refined clusters:
   - Use the mean vector of all documents within each cluster.
2. Optimize cluster assignments based on recalculated centers.

---

### **8. Refinement of Clusters**
**Purpose:** Further improve cluster coherence and interpretability.

**Process:**
- Reassign documents based on similarity to cluster centroids.
- Merge or split clusters as needed based on content analysis.

**Output:**
- Refined cluster assignments and centroids.

---

### **9. Label Assignment**
**Purpose:** Assign meaningful, concise labels to clusters.

**Process:**
1. Extract top terms for each cluster from the DFM.
2. Use ChatGPT to suggest labels based on the top terms.
3. Assign labels to clusters for interpretability.

**Output:** A labeled clustering model.

---

### **10. Saving Models for SectorInsightRv2**
**Purpose:** Save the key outputs for integration into the **SectorInsightRv2** package.

#### **Steps:**
1. Save the **Document-Feature Matrix (DFM):**
   - File: `output/models/dfm_reduced.rds`
   - Description: Contains the top 1000 terms for clustering and analysis.

2. Save the **K-means Model:**
   - File: `output/models/kmeans_model.rds`
   - Description: Includes cluster assignments and centroids.

3. Save additional outputs:
   - Refined clusters using LDA for sub-clustering.
   - Extracted linguistic features (nouns and verbs) from `udpipe` for enriched analysis.

4. Maintain a version-controlled directory structure for model updates and reuse.

**Output:**
- Two primary models (`dfm_reduced.rds` and `kmeans_model.rds`) and auxiliary data for advanced analysis are prepared for deployment in SectorInsightRv2.

---

## **Integration with SectorInsightRv2**

### **Usage of DFM and K-means Models**
1. **DFM for Topic Analysis:**
   - The DFM can be loaded into SectorInsightRv2 to analyze term distributions across topics.
   - Enables visualization of term importance and facilitates keyword-based filtering.

2. **K-means Model for Classification:**
   - Use the K-means model to classify new documents into existing clusters.
   - Leverage centroids to identify the closest topic for a given document.

#### **Example:**
```r
# Load saved models
kmeans_model <- readRDS("output/models/kmeans_model.rds")
dfm <- readRDS("output/models/dfm_reduced.rds")

# Predict cluster for a new document
new_document <- "Exploring renewable energy solutions for rural communities."
tokenized_doc <- quanteda::tokens(new_document, ngrams = 1:3)
tfidf_doc <- quanteda::dfm(tokenized_doc, tolower = TRUE)

# Match with existing clusters
cluster_assignment <- predict(kmeans_model, newdata = tfidf_doc)
cat("Assigned Cluster:", cluster_assignment)
```

---

## **Optimization Strategies**

### **Chunking for Scalability**
- Divide data into smaller chunks for efficient processing.
- Process chunks in parallel to maximize resource utilization.

### **Dimensionality Reduction**
- Retain only the top 1000 terms based on TF-IDF scores.
- Use these terms to generate a reduced DFM for clustering.

### **Dynamic Refinement**
- Apply LDA to refine low-coherence clusters.
- Recalculate cluster centers iteratively to ensure stability.

---

## **Key Outputs**

### **Document-Feature Matrix**
- Sparse matrix with the top 1000 terms.
- Ready for clustering and sub-clustering.

### **K-means Model**
- Cluster assignments and centroids.
- Refined through LDA and recalculated centroids.

### **Cluster Labels**
- Concise and interpretable labels generated via ChatGPT.

### **SectorInsightRv2 Integration**
- The **DFM** and **K-means model** are designed for seamless integration into SectorInsightRv2, providing reusable, scalable solutions for sector-based insights.

---

## **Reproducibility Guidance**

1. **Version Control:**
   - Track all changes using Git.
2. **Environment Setup:**
   - Ensure required R libraries are installed.
3. **Random Seeds:**
   - Set random seeds for clustering to ensure consistent results.
4. **Testing:**
   - Validate the pipeline with a small dataset before scaling up.
