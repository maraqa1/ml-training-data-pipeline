
library(quanteda)      # Text analysis and dfm processing
library(Matrix)        # Sparse matrix operations
library(topicmodels)   # LDA topic modeling
library(textmineR)     # Text mining and coherence score calculation
library(readr)         # Reading CSV files
library(httr)          # API calls (for GPT-based cluster labeling)
library(jsonlite)      # JSON processing



#1. Model Conversion & Validation
#' Convert Models for Processing
#'
#' Converts the TF-IDF model (dfm) to a dgCMatrix (sparse matrix) and removes empty documents.
#'
#' @param tfidf_model A TF-IDF model (dfm or dgCMatrix) loaded from an RDS file.
#' @param kmeans_model A kmeans object with a numeric vector in the 'cluster' field.
#' @return A list containing:
#'   \item{tfidf}{A dgCMatrix of the TF-IDF model with empty rows removed.}
#'   \item{kmeans}{The kmeans object with updated cluster assignments.}
convertModelsForProcessing <- function(tfidf_model, kmeans_model) {
  message("🔹 Running model conversions and validation...")
  
  # Ensure TF-IDF is in dgCMatrix format
  if (!inherits(tfidf_model, "dgCMatrix")) {
    message("⚠️ Converting TF-IDF model to sparse matrix format...")
    tfidf_model <- as(tfidf_model, "dgCMatrix")
  }
  message("✅ TF-IDF model is in `dgCMatrix` format.")
  
  # Remove empty rows (documents with zero term frequencies)
  non_empty <- rowSums(tfidf_model) > 0
  tfidf_model <- tfidf_model[non_empty, ]
  kmeans_model$cluster <- kmeans_model$cluster[non_empty]
  
  message("✅ Empty rows removed. New TF-IDF dimensions: ", paste(dim(tfidf_model), collapse = " x "))
  
  return(list(tfidf = tfidf_model, kmeans = kmeans_model))
}

# --- Testing convertModelsForProcessing ---
test_convertModels <- function() {
  tfidf_model <- readRDS("tfidf_model.rds")  # Replace with your sample file
  kmeans_model <- readRDS("kmeans_model.rds")  # Replace with your sample file
  result <- convertModelsForProcessing(tfidf_model, kmeans_model)
  
  message("Dimensions of converted TF-IDF: ", paste(dim(result$tfidf), collapse = " x "))
  message("Number of cluster assignments: ", length(result$kmeans$cluster))
}
# Uncomment to run the test:
# test_convertModels()


#2. Preprocessing the Cluster Matrix for LDA
#' Preprocess Cluster Matrix for LDA
#'
#' Converts a quanteda dfm for a cluster to a sparse dgCMatrix with numeric term counts.
#'
#' @param cluster_dfm A dfm object (subset of the TF-IDF model) for one cluster.
#' @return A dgCMatrix suitable for LDA. Returns NULL if all rows are empty.
preprocessClusterMatrix <- function(cluster_dfm) {
  library(Matrix)
  message("🔹 Converting TF-IDF to LDA-compatible format...")
  
  # Ensure the dfm's nonzero values are numeric
  message("   🔎 Before conversion: class(cluster_dfm@x) = ", class(cluster_dfm@x))
  cluster_dfm@x <- as.numeric(cluster_dfm@x)
  message("   ✅ After conversion: class(cluster_dfm@x) = ", class(cluster_dfm@x))
  
  # Directly convert the dfm to a sparse dgCMatrix
  cluster_matrix <- as(cluster_dfm, "dgCMatrix")
  
  # Debug: output structure of the matrix
  message("📌 Checking cluster_matrix Structure:")
  print(str(cluster_matrix))
  
  # Ensure values are numeric (they should be, but we double-check)
  cluster_matrix@x <- as.numeric(round(cluster_matrix@x))
  
  # Remove empty rows (documents with no terms)
  non_empty <- rowSums(as.matrix(cluster_matrix)) > 0
  if (sum(non_empty) == 0) {
    message("⚠️ Cluster is empty after filtering. Returning NULL.")
    return(NULL)
  }
  cluster_matrix <- cluster_matrix[non_empty, , drop = FALSE]
  
  message("✅ Preprocessing completed. Final matrix dimensions: ", paste(dim(cluster_matrix), collapse = " x "))
  return(cluster_matrix)
}

# --- Testing preprocessClusterMatrix ---
test_preprocessClusterMatrix <- function() {
  # Load a sample dfm (replace with your file or object)
  dfm_sample <- readRDS("sample_cluster_dfm.rds")
  cluster_matrix <- preprocessClusterMatrix(dfm_sample)
  
  if (is.null(cluster_matrix)) {
    message("Test failed: No valid documents after preprocessing.")
  } else {
    message("Test passed: Cluster matrix dimensions: ", paste(dim(cluster_matrix), collapse = " x "))
  }
}
# Uncomment to run the test:
# test_preprocessClusterMatrix()

#3. Compute Coherence Score for Each Cluster
#' Compute Coherence Score Per Cluster
#'
#' For each cluster (given by cluster_labels), this function applies LDA on the cluster's
#' document-term matrix and computes the coherence score.
#'
#' @param tfidf_model A dgCMatrix TF-IDF model.
#' @param cluster_labels A numeric vector of cluster assignments (one per document).
#' @param lda_topics Integer specifying the number of LDA topics.
#' @return A numeric vector of average coherence scores, named by cluster.
computeCoherenceScore <- function(tfidf_model, cluster_labels=kmeans_model$cluster, lda_topics = 5) {
  library(topicmodels)
  library(Matrix)
  library(textmineR)
  
  message("🔹 Computing coherence score...")
  
  unique_clusters <- unique(cluster_labels)
  coherence_scores <- numeric(length(unique_clusters))
  names(coherence_scores) <- unique_clusters
  
  for (cluster_id in unique_clusters) {
    message("\n📊 Checking Cluster ", cluster_id, " for coherence...")
    
    # Subset TF-IDF matrix for the cluster
    cluster_indices <- which(cluster_labels == cluster_id)
    cluster_dfm <- tfidf_model[cluster_indices, ]
    
    if (nrow(cluster_dfm) < 10) {
      message("⚠️ Cluster ", cluster_id, " is too small for coherence evaluation. Skipping...")
      next
    }
    
    message("🔍 DEBUG: Checking cluster_dfm structure BEFORE processing...")
    print(str(cluster_dfm))
    
    # ✅ Step 1: Preprocess matrix for LDA
    cluster_matrix <- preprocessClusterMatrix(cluster_dfm)
    if (is.null(cluster_matrix)) {
      message("⚠️ Skipping coherence calculation for Cluster ", cluster_id, " due to empty matrix.")
      next
    }
    
    message("✅ Matrix processed successfully!")
    
    # ✅ Step 2: Apply LDA
    message("🔹 Applying LDA on Cluster ", cluster_id, "...")
    lda_model <- LDA(cluster_matrix, k = lda_topics, control = list(seed = 1234))
    
    if (is.null(lda_model)) {
      message("❌ LDA failed for Cluster ", cluster_id, ". Skipping...")
      next
    }
    
    message("✅ LDA applied successfully!")
    
    # ✅ Step 3: Ensure Vocabulary Consistency
    message("🔹 Checking word_matrix Structure BEFORE coherence calculation...")
    word_matrix <- lda_model@beta  # Topic-word distribution from LDA
    dtm_vocab <- colnames(cluster_matrix)  # Vocabulary from dtm
    lda_vocab <- colnames(word_matrix)     # Vocabulary from LDA
    
    # ✅ Intersect both vocabularies
    common_vocab <- intersect(dtm_vocab, lda_vocab)
    if (length(common_vocab) < 2) {
      message("⚠️ Very few common terms between LDA and TF-IDF matrix. Skipping...")
      next
    }
    
    # ✅ Subset both matrices to have the same vocabulary
    word_matrix <- word_matrix[, common_vocab, drop = FALSE]
    cluster_matrix <- cluster_matrix[, common_vocab, drop = FALSE]
    
    # ✅ Step 4: Compute coherence score
    tryCatch({
      coherence_values <- CalcProbCoherence(word_matrix, cluster_matrix)
      avg_coherence <- mean(coherence_values, na.rm = TRUE)
      
      coherence_scores[cluster_id] <- avg_coherence
      message("✅ Coherence Score for Cluster ", cluster_id, ": ", round(avg_coherence, 4))
      
    }, error = function(e) {
      message("❌ ERROR computing coherence score for Cluster ", cluster_id, ": ", e$message)
    })
  }
  
  return(coherence_scores)
}

# --- Testing computeCoherenceScore ---
test_computeCoherenceScore <- function() {
  tfidf_model <- readRDS("tfidf_model.rds")  # Use a sample file
  kmeans_model <- readRDS("kmeans_model.rds")  # Use a sample file
  
  # Convert models
  models <- convertModelsForProcessing(tfidf_model, kmeans_model)
  tfidf_model <- models$tfidf
  cluster_labels <- kmeans_model$cluster
  
  coherence <- computeCoherenceScore(tfidf_model, cluster_labels, lda_topics = 5)
  print(coherence)
}
# Uncomment to run the test:
# test_computeCoherenceScore()

# 3.1
#' Recalculate K-Means cluster centroids after LDA-based splitting.
#'
#' @param tfidf_model A `dgCMatrix` representing the TF-IDF matrix (documents × features).
#' @param new_clusters A numeric vector of updated cluster assignments.
#' @return A matrix of recalculated cluster centroids (num_clusters × num_features).
#' @examples
#' new_centers <- recomputeKMeansCenters(tfidf_model, new_clusters)
recomputeKMeansCenters <- function(tfidf_model, cluster_assignments) {
  message("🔹 Recomputing K-Means centroids...")
  
  unique_clusters <- unique(cluster_assignments)
  num_clusters <- length(unique_clusters)
  num_features <- ncol(tfidf_model)
  
  new_centers <- matrix(0, nrow = num_clusters, ncol = num_features)
  rownames(new_centers) <- as.character(unique_clusters)
  colnames(new_centers) <- colnames(tfidf_model)
  
  for (cluster_id in unique_clusters) {
    cluster_indices <- which(cluster_assignments == cluster_id)
    
    if (length(cluster_indices) > 0) {
      new_centers[as.character(cluster_id), ] <- colMeans(tfidf_model[cluster_indices, , drop = FALSE])
    }
  }
  
  message("✅ New K-Means centroids computed.")
  
  # Create new k-means model object
  updated_kmeans <- list(
    cluster = cluster_assignments,
    centers = new_centers,
    totss = NA,
    withinss = NA,
    tot.withinss = NA,
    betweenss = NA,
    size = table(cluster_assignments),
    iter = NA,
    ifault = NA
  )
  
  class(updated_kmeans) <- "kmeans"
  return(updated_kmeans)
}






#4. LDA-Based Cluster Splitting

#' Split low-coherence clusters using LDA and update K-Means model.
#'
#' @param tfidf_model A `dgCMatrix` representing the TF-IDF matrix.
#' @param kmeans_model A `kmeans` object containing the clustering model.
#' @param lda_topics The number of topics for LDA (default: 5).
#' @param coherence_threshold Minimum coherence score to keep a cluster (default: 0.07).
#' @return An updated `kmeans` model with new cluster assignments and recalculated centroids.
#' @examples
#' updated_kmeans <- splitClustersWithLDA(tfidf_model, kmeans_model, 5, 0.07)
#' Split low-coherence clusters using LDA and update K-Means model.
#'
#' @param tfidf_model A `dgCMatrix` representing the TF-IDF matrix.
#' @param kmeans_model A `kmeans` object containing the clustering model.
#' @param lda_topics The number of topics for LDA (default: 5).
#' @param coherence_threshold Minimum coherence score to keep a cluster (default: 0.07).
#' @return An updated `kmeans` model with new cluster assignments and recalculated centroids.
#' @examples
#' updated_kmeans <- splitClustersWithLDA(tfidf_model, kmeans_model, 5, 0.07)
splitClustersWithLDA_old <- function(tfidf_model, kmeans_model, lda_topics = 5, coherence_threshold = 0.07) {
  library(topicmodels)
  library(Matrix)
  library(textmineR)
  
  message("🔹 Splitting Clusters with LDA (DEBUG MODE)...")
  
  new_clusters <- kmeans_model$cluster  # Copy existing cluster assignments
  max_cluster_id <- max(new_clusters)  # Get max current cluster ID
  
  unique_clusters <- unique(kmeans_model$cluster)
  
  for (cluster_id in unique_clusters) {
    message("\n📊 Checking Cluster ", cluster_id, " for splitting...")
    
    cluster_indices <- which(kmeans_model$cluster == cluster_id)
    cluster_dfm <- tfidf_model[cluster_indices, ]
    
    if (nrow(cluster_dfm) < 10) {
      message("⚠️ Cluster ", cluster_id, " is too small for LDA splitting. Skipping...")
      next
    }
    
    message("📌 Cluster ", cluster_id, " - Document Count: ", nrow(cluster_dfm))
    
    # **Step 1: Preprocessing**
    cluster_matrix <- preprocessClusterMatrix(cluster_dfm)
    if (is.null(cluster_matrix)) next  # Skip if preprocessing failed
    
    # **Step 2: Apply LDA**
    lda_model <- LDA(cluster_matrix, k = lda_topics, control = list(seed = 1234))
    if (is.null(lda_model)) next  # Skip if LDA failed
    
    # **Step 3: Compute Coherence & Split**
    topic_assignments <- topics(lda_model)  # Assign docs to topics
    
    for (topic_id in unique(topic_assignments)) {
      max_cluster_id <- max_cluster_id + 1  # Increment cluster ID
      topic_indices <- cluster_indices[topic_assignments == topic_id]
      new_clusters[topic_indices] <- max_cluster_id  # Assign new cluster ID
    }
    
    message("✅ Cluster ", cluster_id, " was split into ", length(unique(topic_assignments)), " sub-clusters.")
  }
  
  message("🔹 Recalculating K-Means centroids after splitting...")
  updated_kmeans <- recomputeKMeansCenters(tfidf_model, new_clusters)
  
  
  #----------------------------------------------------------------------------
  for (cluster_id in unique(kmeans_model$cluster)) {
    cluster_indices <- which(kmeans_model$cluster == cluster_id)
    cluster_dfm <- tfidf_model[cluster_indices, ]
    
    if (nrow(cluster_dfm) < 10) {
      message("⚠️ Cluster ", cluster_id, " is too small for LDA splitting. Skipping...")
      next
    }
    
    cluster_matrix <- preprocessClusterMatrix(cluster_dfm)
    if (is.null(cluster_matrix)) next  
    
    lda_model <- LDA(cluster_matrix, k = lda_topics, control = list(seed = 1234))
    if (is.null(lda_model)) next  
    
    topic_assignments <- topics(lda_model)  
    
    for (topic_id in unique(topic_assignments)) {
      max_cluster_id <- max(unique(kmeans_model$cluster)) + 1  # Get next valid ID
      topic_indices <- cluster_indices[topic_assignments == topic_id]
      new_clusters[topic_indices] <- max_cluster_id  # Assign proper IDs
    }
  }
  
  # 🚀 Ensure compact numbering to avoid cluster explosion
  new_clusters <- as.numeric(factor(new_clusters))
  updated_kmeans$cluster <- new_clusters
  
  
  message("✅ Labels successfully mapped after cluster splitting.")

  
  #---------------------------------------------------------------------------
  
  return(updated_kmeans)
}

splitClustersWithLDA_old_latest <- function(tfidf_model, kmeans_model, lda_topics = 5, coherence_threshold = 0.07) {
  library(topicmodels)
  library(Matrix)
  library(textmineR)
  
  message("🔹 Splitting Clusters with LDA...")
  
  new_clusters <- kmeans_model$cluster  # Copy existing cluster assignments
  max_cluster_id <- max(new_clusters)  # Get max current cluster ID
  
  unique_clusters <- unique(kmeans_model$cluster)
  
  for (cluster_id in unique_clusters) {
    message("\n📊 Checking Cluster ", cluster_id, " for splitting...")
    
    cluster_indices <- which(kmeans_model$cluster == cluster_id)
    cluster_dfm <- tfidf_model[cluster_indices, , drop = FALSE]  # Ensure matrix format
    
    if (nrow(cluster_dfm) < 10) {
      message("⚠️ Cluster ", cluster_id, " is too small for LDA splitting. Skipping...")
      next
    }
    
    message("📌 Cluster ", cluster_id, " - Document Count: ", nrow(cluster_dfm))
    
    # **Step 1: Preprocessing**
    cluster_matrix <- preprocessClusterMatrix(cluster_dfm)
    if (is.null(cluster_matrix)) next  # Skip if preprocessing failed
    
    # **Step 2: Apply LDA**
    lda_model <- LDA(cluster_matrix, k = lda_topics, control = list(seed = 1234))
    if (is.null(lda_model)) next  # Skip if LDA failed
    
    # **Step 3: Assign New Cluster IDs**
    topic_assignments <- topics(lda_model)  # Get topic-based assignments
    
    for (topic_id in unique(topic_assignments)) {
      max_cluster_id <- max_cluster_id + 1  # Increment safely
      topic_indices <- cluster_indices[topic_assignments == topic_id]
      new_clusters[topic_indices] <- max_cluster_id  # Assign new cluster ID
    }
    
    message("✅ Cluster ", cluster_id, " was split into ", length(unique(topic_assignments)), " sub-clusters.")
  }
  
  message("🔹 Recalculating K-Means centroids after splitting...")
  updated_kmeans <- recomputeKMeansCenters(tfidf_model, new_clusters)
  
  updated_kmeans$cluster <- new_clusters  # Ensure correct mapping
  
  message("✅ Cluster assignments successfully updated after splitting.")
  
  return(updated_kmeans)
}



# --- Testing splitClustersWithLDA ---
test_splitClustersWithLDA <- function() {
  tfidf_model <- readRDS("tfidf_model.rds")  # Replace with sample file
  kmeans_model <- readRDS("kmeans_model.rds")  # Replace with sample file
  
  models <- convertModelsForProcessing(tfidf_model, kmeans_model)
  tfidf_model <- models$tfidf
  kmeans_model <- models$kmeans
  
  new_clusters <- splitClustersWithLDA(tfidf_model, kmeans_model, lda_topics = 5, coherence_threshold = 0.07)
  message("New cluster assignments: ")
  print(table(new_clusters))
}
# Uncomment to run the test:
# test_splitClustersWithLDA()
#--------------------------------------------------------------------------------
#' Split Low-Coherence Clusters Using LDA (Refined Strategy)
#' 
#' Cluster Splitting Logic (Coherence Strategy):

#' This pipeline uses a coherence-based strategy to identify clusters that are poorly defined (i.e., low topic coherence) and then selectively applies LDA-based splitting. To avoid unnecessary fragmentation, only clusters that meet both of the following conditions are split:
#'  Coherence score is below 0.07
#' 
#' Cluster contains 10 or more documents
#
#' This strategy ensures meaningful refinement of noisy or mixed clusters while preserving small or already coherent clusters. It balances interpretability with quality of topic separation.
#'
#' This function evaluates clusters using topic coherence. Low-coherence clusters
#' are considered for splitting only if they have enough documents to support meaningful topic separation.
#'
#' @param tfidf_model A `dgCMatrix` representing the TF-IDF matrix.
#' @param kmeans_model A `kmeans` object containing the clustering model.
#' @param lda_topics Integer: number of topics for LDA. Default is 5.
#' @param coherence_threshold Numeric: coherence score below which clusters are eligible for splitting. Default is 0.07.
#' @param min_docs_for_split Integer: minimum number of documents in a cluster to attempt LDA-based splitting. Default is 10.
#' @return An updated `kmeans` model with new cluster assignments and recalculated centroids.
#'
#' @details
#' This refined logic ensures that only low-coherence clusters with sufficient size are split. Very small or noisy clusters are preserved as-is to avoid artificial fragmentation.
#'
#' Coherence is calculated using `textmineR::CalcProbCoherence`, and LDA is applied
#' only when a cluster has both low coherence and enough documents to support
#' stable topic modeling.
#'
#' @examples
#' updated_kmeans <- splitClustersWithLDA(tfidf_model, kmeans_model, lda_topics = 5, coherence_threshold = 0.07)
splitClustersWithLDA <- function(tfidf_model, kmeans_model, lda_topics = 5, coherence_threshold = 0.07, min_docs_for_split = 10) {
  library(topicmodels)
  library(Matrix)
  library(textmineR)
  
  message("🔹 Splitting low-coherence clusters (Refined Strategy)...")
  
  # Step 1: Compute coherence scores for all clusters
  cluster_labels <- kmeans_model$cluster
  coherence_scores <- computeCoherenceScore(tfidf_model, cluster_labels, lda_topics)
  
  # Step 2: Identify candidates for splitting
  candidate_clusters <- names(which(coherence_scores < coherence_threshold))
  
  message("📌 Candidates for splitting (low coherence): ", paste(candidate_clusters, collapse = ", "))
  
  new_clusters <- cluster_labels
  max_cluster_id <- max(cluster_labels)
  
  for (cluster_id in candidate_clusters) {
    cluster_id <- as.integer(cluster_id)
    cluster_indices <- which(cluster_labels == cluster_id)
    
    if (length(cluster_indices) < min_docs_for_split) {
      message("⚠️ Skipping Cluster ", cluster_id, ": too few documents (", length(cluster_indices), ")")
      next
    }
    
    message("\n🔍 Splitting Cluster ", cluster_id, " with ", length(cluster_indices), " documents...")
    
    # Subset and preprocess
    cluster_dfm <- tfidf_model[cluster_indices, , drop = FALSE]
    cluster_matrix <- preprocessClusterMatrix(cluster_dfm)
    if (is.null(cluster_matrix)) {
      message("⚠️ Preprocessing failed. Skipping Cluster ", cluster_id)
      next
    }
    
    # Apply LDA
    lda_model <- LDA(cluster_matrix, k = lda_topics, control = list(seed = 1234))
    topic_assignments <- topics(lda_model)
    
    for (topic_id in unique(topic_assignments)) {
      max_cluster_id <- max_cluster_id + 1
      topic_docs <- cluster_indices[topic_assignments == topic_id]
      new_clusters[topic_docs] <- max_cluster_id
    }
    
    message("✅ Cluster ", cluster_id, " split into ", length(unique(topic_assignments)), " sub-clusters.")
  }
  
  # Step 3: Reassign compact IDs
  new_clusters <- as.numeric(factor(new_clusters))
  updated_kmeans <- recomputeKMeansCenters(tfidf_model, new_clusters)
  updated_kmeans$cluster <- new_clusters
  
  message("✅ Clustering updated after splitting.")
  return(updated_kmeans)
}



#------------------------------------------------------------------------------

library(quanteda)

#' Extract Top Terms Per Cluster
#'
#' This function identifies the most important terms for each cluster directly
#' from a dgCMatrix, using appropriate methods to handle sparse matrices efficiently.
#'
#' @param dfm A document-feature matrix (DFM) or a compatible matrix (e.g., dgCMatrix).
#' @param cluster_assignments A numeric vector indicating cluster assignments for each document.
#' @param top_n The number of top terms to extract per cluster.
#' @return A named list where each cluster ID maps to a vector of top terms.

extractClusterTerms <- function(dfm, cluster_assignments, top_n = 10) {
  library(quanteda)
  message("🔹 Extracting Top Terms Per Cluster...")
  cluster_terms <- list()
  unique_clusters <- sort(unique(cluster_assignments))
  
  for (cluster_id in unique_clusters) {
    message(paste("Processing Cluster", cluster_id))
    cluster_indices <- which(cluster_assignments == cluster_id)
    
    if (length(cluster_indices) == 0) {
      message(paste("⚠️ Cluster", cluster_id, "has no documents. Skipping..."))
      next
    }
    
    # Subset the dfm for the current cluster
    cluster_dfm <- dfm[cluster_indices, , drop = FALSE]
    
    # Check if there are documents to process
    if (nrow(cluster_dfm) == 0) {
      message(paste("⚠️ Cluster", cluster_id, "is empty after subsetting. Skipping..."))
      next
    }
    
    # Extract top terms directly using matrix operations
    term_sums <- Matrix::colSums(cluster_dfm)
    top_terms_indices <- order(term_sums, decreasing = TRUE)[1:min(top_n, length(term_sums))]
    top_terms <- colnames(cluster_dfm)[top_terms_indices]
    
    cluster_terms[[as.character(cluster_id)]] <- top_terms
    message(paste("✅ Cluster", cluster_id, "- Top Terms:", paste(top_terms, collapse = ", ")))
  }
  
  return(cluster_terms)
}


# --- Testing extractClusterTerms ---
library(Matrix)
library(quanteda)

# --- Testing extractClusterTerms ---


# --- Updated Test Function for extractClusterTerms ---
test_extractClusterTerms <- function() {
  library(Matrix)
  library(quanteda)
  # Sample documents
  documents <- c("finance investment risk market",
                 "health medicine treatment disease",
                 "technology innovation AI machine learning")
  
  # Create DFM using quanteda
  tokens_sample <- tokens(documents, remove_punct = TRUE)
  dfm_sample <- dfm(tokens_sample, tolower = TRUE)
  
  # Convert dfm to dgCMatrix
  matrix_sample <- as(dfm_sample, "dgCMatrix")
  
  # Define mock cluster assignments
  sample_clusters <- c(1, 2, 3)  # Each document is its own cluster for testing
  
  # Run the function
  terms <- extractClusterTerms(matrix_sample, sample_clusters, top_n = 5)
  
  # Output
  print(terms)
}

# Run the test
#test_extractClusterTerms()




#' Generate Cluster Labels using GPT
#'
#' Queries OpenAI API to generate human-readable cluster labels.
#'
#' @param cluster_terms A named list where each element is a vector of top terms.
#' @param api_key A string containing the OpenAI API key.
#' @return A named list with cluster labels.
generateClusterLabels <- function(cluster_terms, api_key) {
  library(httr)
  library(jsonlite)
  
  api_url <- "https://api.openai.com/v1/chat/completions"
  cluster_labels <- list()
  
  for (cluster_id in names(cluster_terms)) {
    prompt <- paste("Generate a short and meaningful label for these keywords: ", paste(cluster_terms[[cluster_id]], collapse = ", "))
    
    response <- POST(
      api_url,
      add_headers(Authorization = paste("Bearer", api_key)),
      content_type_json(),
      body = toJSON(list(
        model = "gpt-4",
        messages = list(
          list(role = "system", content = "You are an expert in topic modeling and clustering."),
          list(role = "user", content = prompt)
        )
      ), auto_unbox = TRUE)
    )
    
    content_response <- content(response, "parsed")
    cluster_labels[[cluster_id]] <- content_response$choices[[1]]$message$content
    message("✅ Cluster ", cluster_id, " Label: ", cluster_labels[[cluster_id]])
  }
  
  return(cluster_labels)
}

# --- Testing generateClusterLabels ---
test_generateClusterLabels <- function() {
  sample_terms <- list(
    "1" = c("finance", "investment", "risk", "market"),
    "2" = c("health", "medicine", "treatment", "disease")
  )
  api_key <- "your_openai_api_key"  # Replace with your actual API key
  labels <- generateClusterLabels(sample_terms, api_key)
  print(labels)
}
# Uncomment to run the test:
# test_generateClusterLabels()

#' Update K-Means Model with Cluster Labels
#'
#' Stores GPT-generated cluster labels inside the K-Means model object.
#'
#' @param kmeans_model A trained kmeans model object.
#' @param cluster_labels A named list of cluster labels.
#' @return The updated kmeans model with labels stored as an attribute.
updateKMeansWithLabels_Old <- function(kmeans_model, cluster_labels) {
  message("🔹 Updating K-Means model with GPT-generated labels...")
  
  # Ensure labels are assigned in the same order as kmeans_model$cluster
  cluster_ids <- unique(kmeans_model$cluster)  # Extract unique clusters
  ordered_labels <- cluster_labels[as.character(cluster_ids)]  # Reorder labels
  
  kmeans_model$labels <- ordered_labels  # Assign to kmeans_model
  
  message("✅ Cluster labels successfully integrated into K-Means model.")
  
  return(kmeans_model)
}

# --- Testing updateKMeansWithLabels ---
test_updateKMeansWithLabels_Old <- function() {
  sample_kmeans <- list(cluster = c(1, 2, 3))
  sample_labels <- list("1" = "Finance & Investment", "2" = "Healthcare")  # Missing label for "3"
  
  updated_kmeans <- updateKMeansWithLabels(sample_kmeans, sample_labels)
  print(updated_kmeans$labels)
}

# Uncomment to run the test:
# test_updateKMeansWithLabels()


#-----------------------------------------------------------------------------

#' Update K-Means Model with Cluster Labels (Document-Level)
#'
#' Adds a document-level topic label vector to the K-Means model object.
#'
#' @param kmeans_model A trained kmeans model object (with $cluster).
#' @param cluster_labels A named list or named character vector of cluster labels (names = cluster numbers).
#' @return The updated K-Means model with a $labels field (same length as number of documents).
updateKMeansWithLabels <- function(kmeans_model, cluster_labels) {
  message("🔹 Updating K-Means model with GPT-generated labels...")
  
  if (is.null(kmeans_model$cluster)) {
    stop("❌ kmeans_model must have a $cluster field.")
  }
  
  cluster_ids <- kmeans_model$cluster
  
  # Assign document-level labels using cluster assignment
  labels_per_document <- cluster_labels[as.character(cluster_ids)]
  
  if (any(is.na(labels_per_document))) {
    warning("⚠️ Some cluster labels are missing for the given cluster numbers.")
  }
  
  kmeans_model$labels <- labels_per_document
  
  message("✅ Cluster labels successfully integrated into K-Means model.")
  
  return(kmeans_model)
}

#' Standardize KMeans Model Labels
#'
#' Converts named cluster labels into a per-document character vector and assigns it to kmeans_model$labels.
#'
#' @param kmeans_model A kmeans model with a named list or vector in $labels (cluster_id -> label).
#' @return The updated kmeans model with standardized per-document labels.
standardize_kmeans_labels <- function(kmeans_model) {
  message("🔧 Converting named cluster labels to per-document topic labels...")
  
  # Step 1: Convert named list to named character vector
  label_lookup <- unlist(kmeans_model$labels)
  
  # Step 2: Ensure cluster assignments are character keys
  cluster_ids <- as.character(kmeans_model$cluster)
  
  # Step 3: Map each document to its cluster label
  document_labels <- label_lookup[cluster_ids]
  
  # Step 4: Check for missing values
  if (any(is.na(document_labels))) {
    warning("⚠️ Some cluster IDs did not have corresponding labels. Consider checking label coverage.")
  }
  
  # Step 5: Assign as per-document character vector (remove names!)
  kmeans_model$labels <- document_labels
  names(kmeans_model$labels) <- NULL
  
  message("✅ Model is now standardized with clean per-document labels.")
  
  return(kmeans_model)
}


#-------------------------------------------------------------------------------

#' Save Updated K-Means and TF-IDF Models
#'
#' Saves the updated models with filenames that include the number of clusters and the current date.
#'
#' @param results A list containing the updated kmeans and tfidf models.
#' @param output_dir The directory where the models should be saved.
#' @return The paths of the saved files.
saveUpdatedModels <- function(results, output_dir = "output/models/kmeans/") {
  message("🔹 Saving updated models...")
  
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  num_clusters <- length(unique(results$kmeans$cluster))  
  current_date <- format(Sys.Date(), "%Y-%m-%d")  
  
  kmeans_filename <- sprintf("%skmeans_%d_clusters_%s.rds", output_dir, num_clusters, current_date)
  tfidf_filename <- sprintf("%stfidf_%d_clusters_%s.rds", output_dir, num_clusters, current_date)
  
  saveRDS(results$kmeans, kmeans_filename)
  saveRDS(results$tfidf, tfidf_filename)
  
  message("✅ Models saved successfully!")
  message("📁 K-Means Model: ", kmeans_filename)
  message("📁 TF-IDF Model: ", tfidf_filename)
  
  return(list(kmeans_path = kmeans_filename, tfidf_path = tfidf_filename))
}


# --- Example Usage ---
# saveUpdatedModels(results)



#5. Main Pipeline Function
#' Run Phase 1 Pipeline
#'
#' Loads the dataset, TF-IDF, K-Means, and embeddings; validates the models; computes coherence scores;
#' splits clusters with low coherence using LDA; and returns updated clustering and coherence scores.
#'
#' @param input_file Path to the CSV dataset.
#' @param kmeans_path Path to the saved kmeans model RDS file.
#' @param tfidf_path Path to the saved TF-IDF model RDS file.
#' @param embedding_path Path to the saved pre-trained embeddings RDS file.
#' @param lda_topics Integer specifying the number of topics for LDA.
#' @param coherence_threshold Numeric threshold for cluster coherence.
#' @param break_clusters Logical flag; if TRUE, low-coherence clusters are split using LDA.
#' @return A list containing:
#'   \item{kmeans}{Updated kmeans model with new cluster assignments.}
#'   \item{tfidf}{The validated TF-IDF matrix (dgCMatrix).}
#'   \item{coherence}{A numeric vector of coherence scores per cluster.}

runPhase1Pipeline <- function(input_file, kmeans_path, tfidf_path, lda_topics = 5, coherence_threshold = 0.07, break_clusters = TRUE, api_key = NULL) {
  message("🔹 [Step 1] Loading Pre-Trained Models...")
  
  dataset <- read_csv(input_file)
  kmeans_model <- readRDS(kmeans_path)
  tfidf_model <- readRDS(tfidf_path)
  
  models <- convertModelsForProcessing(tfidf_model, kmeans_model)
  tfidf_model <- models$tfidf
  kmeans_model <- models$kmeans
  
  message("🔹 [Step 2] Computing Coherence Score Per Cluster...")
  coherence_scores <- computeCoherenceScore(tfidf_model, kmeans_model$cluster)
  
  # ⚠️ Fix: Only split clusters with low coherence
  low_coherence_clusters <- which(unlist(coherence_scores) < coherence_threshold)
  
  if (length(low_coherence_clusters) > 0 && break_clusters) {
    message("⚠️ Some clusters have low coherence! Splitting them using LDA...")
    
    # ✅ Call the fixed split function
    kmeans_model <- splitClustersWithLDA(tfidf_model, kmeans_model, lda_topics, coherence_threshold)
    
    # ✅ Compute coherence again after splitting
    message("🔹 Recomputing coherence after splitting...")
    coherence_scores <- computeCoherenceScore(tfidf_model, kmeans_model$cluster)
  }
  
  # 🔹 [Step 3] Extract Cluster Terms
  message("🔹 Extracting top terms for cluster labeling...")
  cluster_terms <- extractClusterTerms(tfidf_model, kmeans_model$cluster, top_n = 50)
  
  # 🔹 [Step 4] Generate GPT Labels (if API Key is provided)
  if (!is.null(api_key)) {
    message("🔹 Generating cluster labels using GPT...")
    cluster_labels <- generateClusterLabels(cluster_terms, api_key)
    
    # 🔹 [Step 5] Update K-Means Model with Labels
    kmeans_model <- updateKMeansWithLabels(kmeans_model, cluster_labels)
    kmeans_model <-standardize_kmeans_labels (kmeans_model)
  }
  kmeans_model$ifault <- 0
  
  # Save the updated models
  saved_paths <- saveUpdatedModels(list(kmeans = kmeans_model, tfidf = tfidf_model))
  
  message("✅ Phase 1 Pipeline Completed Successfully!")
  return(list(kmeans = kmeans_model, tfidf = tfidf_model, coherence = coherence_scores))
}




# Run the pipeline with the correct file paths
results <- runPhase1Pipeline(
  input_file = "C:/R_Home/UK_Innovate_topic_modelling/dirty_work/consolidated_document_with_sector_1st_complete_02_09_2024.csv",
  kmeans_path = "output/models/dfm/27_new_kmeans_model_udpipe_11_02_25.rds",
  tfidf_path = "output/models/kmeans/27_tfidf_reduced_udpipe_11_02_25.rds",
 
 # kmeans_path = "output/models/kmeans/19_new_kmeans_model_udpipe_23_03_25.rds",
  #tfidf_path = "output/models/dfm/19_tfidf_reduced_udpipe_23_03_25.rds",
  
  #embedding_path = "output/models/pre_trained/pretrained_embeddings.rds",
  lda_topics = 5,
  coherence_threshold = 0.07,
  break_clusters = TRUE,
  api_key
  
)

print(results$kmeans$labels)
