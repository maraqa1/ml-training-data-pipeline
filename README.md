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

**Output:** A dataframe with `Domain`, `Level 1`, `Level 2`, `Primary Sector`, `Secondary Sector`. Then `combined_features` is the concatenation of the "hierarchical topics" and "Identified sectors" which is used for clustering.

---

### **5. Building the Document-Feature Matrix (DFM)**

**Purpose:** Use the consolidated API results to create a unified Document-Feature Matrix (DFM) for clustering.

#### **Steps:**
1. Use the consolidated results from the API to generate the DFM:
   - Tokenize `combined_features` into unigrams, bigrams, and trigrams.
   - Extract nouns and verbs from `combined_features` using the `udpipe` model. These linguistic features are stored in the `cleaned_text_1` column.
   - Apply TF-IDF (Term Frequency-Inverse Document Frequency) weighting to emphasize distinctive terms and down-weight common, less-informative terms
2. Retain the top 1000 terms with the highest TF-IDF scores across the entire dataset:
   - Compute column sums of the DFM to identify frequent terms.
   - Filter terms to create a reduced DFM.

**Advantages:**
- Focuses on the most relevant terms for clustering.
- Reduces computational load while maintaining meaningful features.

---

### 6. Initial Clustering with K-means

**Purpose:** To efficiently group documents into broad clusters using K-means, establishing a scalable and computationally efficient baseline for further refinement. This step creates hard cluster assignments and identifies general topics, which are then refined in subsequent steps using LDA.

---

#### Rationale for Using K-means First
1. **Scalability and Efficiency:**
   - K-means is computationally efficient, making it suitable for large datasets and high-dimensional TF-IDF matrices.
   - It works well in cases where the data is numerical (e.g., reduced TF-IDF features) and the clusters are expected to be spherical in the feature space.

2. **Hard Clustering for Clear Assignments:**
   - K-means provides discrete cluster assignments, which are useful for downstream steps like coherence evaluation and centroid-based analysis.
   - Unlike soft clustering methods (e.g., LDA), K-means ensures every document is assigned to exactly one cluster, simplifying the refinement process.

3. **Baseline Grouping for Refinement:**
   - K-means provides a rough but effective first-pass grouping of documents into general clusters.
   - These initial clusters act as a foundation for LDA-based sub-clustering, which addresses specific cases like mixed or incoherent clusters.

4. **Interpretability and Centroids:**
   - K-means produces centroids that represent the central term distributions of each cluster, making it easy to extract top terms and analyze cluster themes.

---

#### Steps

##### Step 6.1: Feature Preparation
1. **Input:** 
   - Processed text data stored in the `cleaned_text_1` column.
   - A Document-Feature Matrix (DFM) with n-grams weighted by TF-IDF.
2. **Process:**
   - Extract the top 1000 terms by column sums from the TF-IDF matrix to reduce dimensionality.
3. **Output:** 
   - A reduced TF-IDF matrix (`tfidf_reduced`) with the most informative terms.

---

##### Step 6.2: K-means Clustering
1. **Input:**
   - The reduced TF-IDF matrix (`tfidf_reduced`).
2. **Process:**
   - Perform K-means clustering on the reduced matrix.
   - Select the number of clusters (`k`) based on evaluation metrics:
     - **Silhouette Score:** Measures intra-cluster similarity and inter-cluster separation.
     - **Elbow Method:** Determines the optimal `k` by finding the point of diminishing returns in explained variance.
     - **Domain Knowledge:** Aligns the choice of `k` with the dataset's characteristics.
   - Assign documents to clusters based on their proximity to cluster centroids.
3. **Output:**
   - **Cluster Assignments:** Each document is assigned to a single cluster.
   - **Cluster Centroids:** Represents the term distributions of each cluster.

---

#### Why Not Use LDA First?
1. **Scalability Challenges:**
   - LDA is computationally expensive, especially for high-dimensional datasets with a large vocabulary size.
2. **Initialization Sensitivity:**
   - LDA requires careful tuning of hyperparameters (e.g., the number of topics, alpha, beta), and poor initialization can result in incoherent topics.
3. **Hard Clustering Requirement:**
   - K-means provides clear, discrete assignments for every document, while LDA's probabilistic output requires additional post-processing to assign documents to a single cluster.

K-means serves as a robust, scalable starting point, while LDA is introduced later to refine clusters.

---

### 7. Post-Clustering Enhancements with LDA

**Purpose:** To refine the initial K-means clusters, improve their coherence, and identify subtopics using Latent Dirichlet Allocation (LDA). This step focuses on handling clusters with mixed or overlapping topics that cannot be resolved through K-means alone.

---

#### Rationale for Using LDA After K-means
1. **Addressing Mixed Clusters:**
   - K-means clusters can sometimes group documents with multiple distinct themes due to limitations in handling text's semantic nuances.
   - LDA models documents as mixtures of topics, allowing it to identify and separate subtopics within clusters.

2. **Improving Coherence:**
   - LDA enhances intra-cluster similarity by assigning documents to refined sub-clusters based on topic probabilities.
   - Coherence scores are used to measure and improve the quality of the topics generated by LDA.

3. **Soft Clustering for Overlaps:**
   - Some documents may belong to multiple topics, and LDA captures this overlap by assigning probabilities for each topic.
   - This flexibility ensures that documents in ambiguous clusters are more accurately categorized.

4. **Focus on Large or Poorly Coherent Clusters:**
   - LDA is computationally more expensive than K-means, so it is selectively applied only to clusters that require further refinement.

---

#### Steps

##### Step 7.1: Cluster Analysis
1. **Input:**
   - Initial clusters from K-means.
2. **Process:**
   - Extract top terms for each cluster from the K-means centroids.
   - Evaluate cluster coherence using:
     - **Cosine Similarity:** Measures intra-cluster similarity.
     - **Coherence Scores:** Based on domain-specific knowledge or word co-occurrence patterns.
   - Identify clusters with low coherence or significant term overlap for refinement.
3. **Output:**
   - Coherence metrics for each cluster.
   - Identification of clusters requiring further refinement.

---

##### Step 7.2: LDA-Based Sub-Clustering
1. **Input:**
   - Low-coherence clusters identified in step 7.1.
   - Reduced TF-IDF matrix (`tfidf_reduced`).
2. **Process:**
   - Apply LDA to refine large or incoherent clusters:
     - Identify subtopics within each low-coherence cluster.
     - Split clusters into sub-clusters if coherence scores fall below a threshold.
   - Reassign documents to new sub-clusters based on LDA's topic probabilities.
   - Recalculate centroids for the new sub-clusters.
3. **Output:**
   - **Refined Cluster Assignments:** Documents are reassigned to sub-clusters.
   - **Updated Centroids:** Represent the new sub-clusters.

---

##### Step 7.3: Recalculate Cluster Centers
1. **Purpose:**
   - To ensure cluster centroids reflect the refined cluster structure after LDA.
2. **Process:**
   - Compute the mean vector of all documents assigned to each sub-cluster.
   - Optimize cluster assignments to maximize intra-cluster similarity.
3. **Output:**
   - Final centroids representing refined clusters.

---

#### Why Combine K-means and LDA?

1. **Efficiency First:** 
   - K-means provides a fast and scalable first-pass clustering.
2. **Refinement Second:**
   - LDA adds semantic depth and resolves issues with poorly coherent or overlapping clusters.
3. **Complementary Strengths:**
   - K-means excels in hard clustering and scalability.
   - LDA captures soft assignments and semantic relationships, enhancing interpretability.

---

#### Final Outputs
1. **Cluster Assignments:**
   - Refined assignments with hard or soft clustering as needed.
2. **Cluster Centroids:**
   - Represent updated term distributions for coherent topics.
3. **Coherence Metrics:**
   - Quantify the quality of final clusters for evaluation and reporting.

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
