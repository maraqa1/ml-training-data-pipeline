# Load necessary libraries
library(tidyverse)
library(tm)
library(textclean)
library(tidytext)
library(text)
library(quanteda)
library(cluster)
library(udpipe)

# Step 1: Load Data
data <- read_csv("C:/R_Home/UK_Innovate_topic_modelling/dirty_work/consolidated_document_with_sector_1st_complete_02_09_2024.csv")

#data <- read_csv("output/consolidated_results.csv")

# Combine features into a single text field for each document
data <- data %>%
  #select(-combined_features) %>%
  #mutate(combined_features = paste(Domain, Level_1, Level_2,Primary_Sector, Secondary_Sector, sep = " "))
  mutate(combined_features = paste(Domain, Level_1, Level_2, subsector, `sub-subsector`, sep = " "))



# Step 2: Text Cleaning Function
clean_text <- function(text) {
  text %>%
    stringr::str_to_lower() %>%
    textclean::replace_contraction() %>%
    #textclean::replace_emoji() %>%
    #textclean::replace_emoticon() %>%
    #textclean::replace_internet_slang() %>%
    tm::removePunctuation() %>%
    tm::removeNumbers() %>%
    tm::stripWhitespace() %>%
    tm::removeWords(c(tm::stopwords("en"), "project", "focuses", "team", "users", 
                      "excuse", "tongue", "sticking", "loring"))
}

# Apply cleaning to the combined features
data <- data %>%
  mutate(cleaned_text = map_chr(combined_features, clean_text))

# ------------------------------------------------------------------------
# 📌 Step 3: Replace spaCy with UDPipe for POS-based Keyword Extraction
# ------------------------------------------------------------------------
# UDPipe is a POS tagging and dependency parsing tool used for extracting
# linguistically meaningful terms (e.g., NOUNs and VERBs) from text.
# It is particularly useful when you want to filter out stopwords, 
# adjectives, and non-content words and keep only the core meaningful units.

# ✅ Official documentation and model download page:
# ➤ https://bnosac.github.io/udpipe/docs/doc0.html

# 📥 Download and load the English model (only once is required per environment)
# The model is downloaded as a `.udpipe` file and then loaded into R

ud_model <- udpipe_download_model(language = "english")
ud_model <- udpipe_load_model(file = ud_model$file_model)

# ------------------------------------------------------------------------
# 📦 Function: Extract Cleaned Noun/Verb Text from a Description Field
# ------------------------------------------------------------------------
# This function:
#   1. Cleans the text by removing punctuation and converting to lowercase.
#   2. Applies UDPipe to annotate it (tokenization, POS tagging, etc.).
#   3. Filters for only NOUNs and VERBs.
#   4. Returns a comma-separated string of unique keywords.

extract_cleaned_text <- function(description) {
  # Preprocess the text
  clean_text <- str_replace_all(description, "[[:punct:]]", "") %>% tolower()
  
  # Annotate the text using udpipe
  annotations <- udpipe_annotate(ud_model, x = clean_text)
  annotations <- as.data.frame(annotations)
  
  # Extract nouns and verbs
  keywords <- annotations %>%
    filter(upos %in% c("NOUN", "VERB")) %>%
    pull(token) %>%
    unique() %>%
    paste(collapse = ", ")
  
  return(keywords)
}

# ------------------------------------------------------------------------
# ⚡️ Step 4: Apply the extraction to an entire dataset using parallel processing
# ------------------------------------------------------------------------
# We're using {furrr} and {progressr} to apply `extract_cleaned_text` 
# efficiently across many documents in parallel, with a progress bar.

library(furrr)
library(progressr)

# Set up progress bar globally (one-time setup)
handlers(global = TRUE)
handlers("txtprogressbar")  # Or use "rstudio" if you're in RStudio

# Use limited number of workers to avoid system overload
plan(multisession, workers = 10)

with_progress({
  # Create progressor (progress bar tracker)
  p <- progressor(along = data$cleaned_text)
  
  data <- data %>%
    mutate(cleaned_text_1 = future_map_chr(cleaned_text, ~ {
      p()  # Advance progress bar
      result <- extract_cleaned_text(.x)
      
      # Safety: always return a single character string
      if (length(result) != 1 || is.null(result)) return(NA_character_)
      result
    }))
})

# Reset back to sequential processing
plan(sequential)


# ------------------------------------------------------------------------
# ✅ Output
# Each row in `data$cleaned_text_1` now contains a string of unique NOUNs and VERBs
# extracted from the corresponding input `cleaned_text`. These can be used to:
#   - Generate cluster/topic labels Or in the future 
#   - Train interpretable NLP models
# ------------------------------------------------------------------------



# Summary and clustering steps will remain the same as they do not depend on spaCy.
data %>% dplyr::select (PublicDescription,cleaned_text_1) %>% head()

#write.csv(data,"output/features.csv") #if already saved
data<- read.csv("output/features.csv")



# Step 3: Extract Summary Themes using TF-IDF with N-grams
extract_summary <- function(text) {
  # Convert to dataframe for tidytext functions
  text_df <- tibble(document = 1, text = text)
  
  # Tokenize with unigrams, bigrams, and trigrams
  tokens <- text_df %>%
    tidytext::unnest_tokens(word, text, token = "ngrams", n = 3) %>%
    dplyr::count(document, word, sort = TRUE) %>%
    tidytext::bind_tf_idf(word, document, n)
  
  # Use a broader set of top features for larger datasets
  top_themes <- tokens %>%
    dplyr::arrange(desc(tf_idf)) %>%
    dplyr::slice_head(n = 10) %>%  # Increase the number of features to expand context
    dplyr::pull(word)
  
  return(paste(top_themes, collapse = " "))
}

# Step 4: Feature Extraction with N-grams
extract_features <- function(df) {
  tokens <- quanteda::tokens(df$cleaned_text_1, ngrams = 1:3)  # Use unigrams, bigrams, and trigrams
  dfm <- quanteda::dfm(tokens)
  dfm <- quanteda::dfm_remove(dfm, quanteda::stopwords("en"))
  tfidf <- quanteda::dfm_tfidf(dfm)
  return(tfidf)
}

# Create a DFM with TF-IDF features
tfidf_features <- extract_features(data)


# Step 5: Reduce DFM Dimensionality
reduce_dfm <- function(dfm, top_n = 2000) {
  top_features <- names(tail(sort(colSums(as.matrix(dfm))), top_n))
  dfm_reduced <- dfm[, top_features, drop = FALSE]  # Keep only the top features
  return(as.matrix(dfm_reduced))  # Convert to matrix for clustering
}

tfidf_reduced <- reduce_dfm(tfidf_features, top_n = 2000)

# Step 6: Clustering to Group Similar Themes
apply_clustering <- function(features, k = 15) {  # Adjust `k` based on dataset size
  set.seed(123)
  kmeans_model <- stats::kmeans(as.matrix(features), centers = k)
  return(kmeans_model)
}

# Apply clustering
kmeans_model_updated <- apply_clustering(tfidf_reduced , k = 15)


# Step 7: Extract Top Terms from K-means Model
extract_top_terms_from_kmeans <- function(kmeans_model, tfidf_matrix, top_n = 20) {
  # Extract cluster centers and term names
  cluster_centers <- kmeans_model$centers
  terms <- colnames(tfidf_matrix)
  
  # Helper function to get top terms for each cluster
  get_top_terms <- function(cluster_center, terms, top_n) {
    top_indices <- order(cluster_center, decreasing = TRUE)[1:top_n]
    top_terms <- terms[top_indices]
    return(paste(top_terms, collapse = ", "))
  }
  
  # Apply to each cluster center and get top terms
  top_terms_per_cluster <- apply(cluster_centers, 1, function(center) {
    get_top_terms(center, terms, top_n)
  })
  
  # Create a data frame for the results
  top_terms_df <- data.frame(
    cluster = 1:nrow(cluster_centers),
    keywords = top_terms_per_cluster,
    stringsAsFactors = FALSE
  )
  
  return(top_terms_df)
}

# Get top terms for each cluster
top_terms_df <- extract_top_terms_from_kmeans(kmeans_model_updated , tfidf_reduced, top_n = 30)

# Print top terms for each cluster
print(top_terms_df)
#-------------------------------------------------------------------------------


# Convert the dataframe content into a formatted string
df_content <- paste0(
  apply(top_terms_df, 1, function(row) {
    paste0("Cluster ", row["cluster"], ": ", row["keywords"])
  }),
  collapse = "\n"
)

# Construct the full ChatGPT prompt
chatgpt_prompt <- paste0(
  "I have a dataset with clusters and associated keywords. The goal is to assign meaningful, concise labels to each cluster based on the provided keywords.\n",
  "Here is the data:\n\n",
  df_content,
  "\n\nEach cluster has a unique set of keywords that describe its theme or topic. Please generate concise and meaningful labels for each cluster, reflecting the overall subject matter."
)

# Print the prompt to verify
cat(chatgpt_prompt)


# Save the processed data
#write_csv(chatgpt_prompt, "output/chatgpt_prompt.csv")

#-------------------------------------------------------------------------------
# TUNE the Clusters using LDA

#-------------------------------------------------------------------------------
#' LDA-Based Sub-Clustering of Low-Coherence K-Means Clusters
#'
#' This function applies Latent Dirichlet Allocation (LDA) to each K-means cluster to evaluate
#' topic coherence. Clusters that demonstrate low semantic coherence are considered noisy or 
#' heterogeneous and are further split into subtopics based on LDA assignments.
#'
#' @param tfidf_reduced A `dgCMatrix` representing the TF-IDF matrix of all documents.
#' @param kmeans_model A K-means model object with a `$cluster` field containing the cluster assignment of each document.
#' @param k Integer. The number of topics to use when fitting LDA within each cluster. Default is 5.
#' @param coherence_threshold Numeric. Minimum average topic coherence required to keep a cluster without splitting. Default is 0.07.
#'
#' @return A numeric vector of updated cluster assignments, where low-coherence clusters are split into subclusters.
#'
#' @details
#' This function is built on NLP best practice that leverages topic coherence to assess the quality
#' of K-means clusters from a semantic perspective. Topic coherence measures how frequently the 
#' top topic words co-occur across documents — a proxy for interpretability and alignment.
#'
#' ## Expert Rationale:
#' - **High Coherence** clusters contain semantically consistent documents and should be preserved.
#' - **Low Coherence** clusters are likely to mix unrelated themes and benefit from sub-topic extraction.
#'
#' The function performs the following steps:
#' 1. Iterates through each K-means cluster.
#' 2. Fits an LDA model using `textmineR::FitLdaModel` to uncover hidden subtopics.
#' 3. Calculates topic coherence using `CalcProbCoherence`.
#' 4. If the average coherence is **below** the defined threshold, the cluster is split.
#'    Otherwise, the cluster is retained as-is.
#'
#' ## Practical Example:
#' - Cluster A contains documents about "battery", "charging", "electric vehicle", "power" — coherence is high ➝ **don't split**.
#' - Cluster B has mixed content like "project", "funding", "animal", "robotics", "cancer" — coherence is low ➝ **split to find clearer topics**.
#'
#' This ensures that final clusters are topically tight, interpretable, and actionable for downstream tasks like labeling, summarization, or prediction.
#'
#' @importFrom textmineR FitLdaModel CalcProbCoherence
#' @importFrom Matrix rowSums
#' @export


#LDA for Textual Sub-clustering
#If working with textual data, you can apply Latent Dirichlet Allocation (LDA) on the subset to identify topics.
split_clusters_with_lda <- function(tfidf_reduced, kmeans_model, k = 5, coherence_threshold = 0.07) {
  library(textmineR)
  library(Matrix)
  
  # Initialize results
  new_clusters <- kmeans_model$cluster
  max_cluster_id <- max(kmeans_model$cluster) # Track the max cluster ID
  
  # Process each K-means cluster
  unique_clusters <- unique(kmeans_model$cluster)
  for (cluster_id in unique_clusters) {
    cat("\nProcessing Cluster:", cluster_id, "\n")
    
    # Subset TF-IDF data for the current cluster
    cluster_indices <- which(kmeans_model$cluster == cluster_id)
    cluster_data <- tfidf_reduced[cluster_indices, ]
    
    # Skip small clusters
    if (nrow(cluster_data) < 10) {
      cat("   Cluster", cluster_id, "is too small for splitting.\n")
      next
    }
    
    
    if (!inherits(cluster_data, "dgCMatrix")) {
      cat("   Converting cluster_data to sparse matrix format...\n")
      cluster_data <- Matrix::sparseMatrix(
        i = as.integer(row(cluster_data)),
        j = as.integer(col(cluster_data)),
        x = as.numeric(cluster_data),
        dims = dim(cluster_data),
        dimnames = dimnames(cluster_data)
      )
    }
    
    
    # Fit LDA
    lda_model <- FitLdaModel(
      cluster_data,
      k = k,
      iterations = 500,
      burnin = 50,
      alpha = 0.1,
      beta = 0.01,
      optimize_alpha = TRUE
    )
    
    # Check coherence of LDA topics
    coherence_scores <- CalcProbCoherence(lda_model$phi, cluster_data)
    avg_coherence <- mean(coherence_scores)
    cat("   Average topic coherence for Cluster", cluster_id, ":", avg_coherence, "\n")
    
    # Decide whether to split - split high coherence clusters
   ## if (avg_coherence < coherence_threshold) {
   ##   cat("   Cluster", cluster_id, "is not cohesive enough for splitting.\n")
   ##   next
  ##  }
    
    if (avg_coherence >= coherence_threshold) {
      cat("   Cluster", cluster_id, "is cohesive and will not be split.\n")
      next
    }
    
    
    # Assign documents to LDA topics
    document_topics <- apply(lda_model$theta, 1, which.max)
    new_topic_ids <- unique(document_topics)
    
    # Split cluster
    for (topic_id in new_topic_ids) {
      max_cluster_id <- max_cluster_id + 1
      topic_indices <- cluster_indices[document_topics == topic_id]
      new_clusters[topic_indices] <- max_cluster_id
    }
    cat("   Cluster", cluster_id, "was split into", length(new_topic_ids), "subclusters.\n")
  }
  
  # Return updated clusters
  return(new_clusters)
}


renumber_clusters <- function(cluster_assignments) {
  # Create a mapping from old cluster IDs to new sequential cluster IDs
  unique_clusters <- sort(unique(cluster_assignments))
  new_cluster_map <- setNames(seq_along(unique_clusters), unique_clusters)
  
  # Apply the mapping to the cluster assignments
  new_cluster_assignments <- unname(new_cluster_map[as.character(cluster_assignments)])
  
  return(new_cluster_assignments)
}

update_kmeans_model <- function(kmeans_model, new_clusters, tfidf_reduced) {
  cat("\nStarting update_kmeans_model...\n")
  
  # Update cluster assignments
  kmeans_model$cluster <- new_clusters
  
  # Extract unique clusters (including new clusters)
  unique_clusters <- sort(unique(new_clusters))
  cat("Unique clusters:", unique_clusters, "\n")
  
  # Recalculate cluster centers for all unique clusters
  new_centers <- list()  # Use a list to collect centers dynamically
  for (cluster_id in unique_clusters) {
    cluster_indices <- which(new_clusters == cluster_id)
    
    cat("Processing cluster:", cluster_id, "with", length(cluster_indices), "documents\n")
    
    if (length(cluster_indices) == 1) {
      # Handle single-document clusters
      cat("   Cluster", cluster_id, "is a single-document cluster. Taking raw vector as center.\n")
      new_centers[[as.character(cluster_id)]] <- as.numeric(tfidf_reduced[cluster_indices, , drop = FALSE])
    } else {
      # Calculate the mean for multi-document clusters
      cat("   Calculating mean for cluster", cluster_id, "\n")
      new_centers[[as.character(cluster_id)]] <- colMeans(tfidf_reduced[cluster_indices, , drop = FALSE])
    }
  }
  
  # Ensure new_centers is a matrix
  new_centers <- do.call(rbind, new_centers)
  cat("New centers dimensions:", dim(new_centers), "\n")
  
  # Update the centers in kmeans_model
  kmeans_model$centers <- new_centers
  
  # Debugging center dimensions
  if (!all(dim(kmeans_model$centers) == c(length(unique_clusters), ncol(tfidf_reduced)))) {
    cat("Mismatch in center dimensions! Expected:", length(unique_clusters), "x", ncol(tfidf_reduced), "\n")
  }
  
  # Update size attribute
  cluster_sizes <- table(new_clusters)
  kmeans_model$size <- as.integer(cluster_sizes)
  cat("Cluster sizes:", kmeans_model$size, "\n")
  
  # Recalculate within-cluster sum of squares
  kmeans_model$tot.withinss <- 0
  for (cluster_id in unique_clusters) {
    cluster_indices <- which(new_clusters == cluster_id)
    center <- kmeans_model$centers[which(unique_clusters == cluster_id), ]
    withinss <- sum(rowSums((tfidf_reduced[cluster_indices, , drop = FALSE] - center)^2))
    kmeans_model$tot.withinss <- kmeans_model$tot.withinss + withinss
    cat("   Within-cluster sum of squares for cluster", cluster_id, ":", withinss, "\n")
  }
  
  cat("\nFinished update_kmeans_model.\n")
  
  # Return updated model
  return(kmeans_model)
}
#------------------------------------------------------------------------------
#' 📚 LDA-Based Sub-Clustering with Adaptive k
#'
#' Splits low-coherence K-means clusters by applying LDA with an adaptive number of topics.
#' Each cluster is split only if its average coherence is below a defined threshold.
#'
#' ## 🔍 Topic Coherence Rationale
#' This function uses `textmineR::CalcProbCoherence()` to compute how semantically tight each topic is,
#' based on word co-occurrence in documents. If coherence is low, the cluster is considered noisy and
#' is split further using LDA topic assignments.
#'
#' ## 🧪 Coherence Calculation in `textmineR`
#' `CalcProbCoherence()`:
#' - Extracts the top N words from each topic in the LDA topic-word matrix (`phi`)
#' - For each word pair (w_i, w_j), it calculates how often they appear together in the same document (TF-IDF or DTM)
#' - Computes log-conditional probabilities and averages them across all pairs
#' - Returns one coherence score per topic
#'
#' Coherence formula (simplified from Mimno et al., 2011):
#' \deqn{C(t) = \sum_{m=2}^{M} \sum_{l=1}^{m-1} \log \frac{D(w_m, w_l) + 1}{D(w_l)}}
#' where D(w_m, w_l) is the number of docs with both words, D(w_l) is the doc count for w_l.
#'
#' ## 📌 In This Function:
#' - For each cluster, try several LDA topic numbers (`k_range`)
#' - Choose the best k based on highest average coherence
#' - If best coherence < threshold → split the cluster by topic assignments
#' - If coherence is high → retain the cluster as is
#'
#' @param tfidf_reduced A `dgCMatrix` sparse TF-IDF matrix of all documents.
#' @param kmeans_model A K-means model object with `$cluster` assignments.
#' @param coherence_threshold Minimum coherence score to retain a cluster without splitting. Default: 0.07.
#' @param k_range Integer vector indicating the candidate topic numbers to evaluate per cluster. Default: 2:6.
#'
#' @return A numeric vector of updated cluster assignments.
#' @references Mimno et al. (2011). Optimizing Semantic Coherence in Topic Models. EMNLP. https://aclanthology.org/D11-1024/
#' @export
split_clusters_with_lda_adaptive <- function(tfidf_reduced, kmeans_model,
                                             coherence_threshold = 0.07,
                                             k_range = 2:6) {
  library(textmineR)
  library(Matrix)
  
  new_clusters <- kmeans_model$cluster
  max_cluster_id <- max(kmeans_model$cluster)
  
  unique_clusters <- unique(kmeans_model$cluster)
  
  for (cluster_id in unique_clusters) {
    cat("\n🔹 Processing Cluster:", cluster_id, "\n")
    
    cluster_indices <- which(kmeans_model$cluster == cluster_id)
    cluster_data <- tfidf_reduced[cluster_indices, , drop = FALSE]
    
    if (nrow(cluster_data) < 10 || ncol(cluster_data) < 5) {
      cat("   ⚠️ Cluster too small to split. Skipping.\n")
      next
    }
    
    cluster_data <- cluster_data[rowSums(cluster_data) > 0, , drop = FALSE]
    if (nrow(cluster_data) < 10) {
      cat("   ⚠️ Cluster became too small after removing empty rows.\n")
      next
    }
    
    best_k <- NULL
    best_score <- -Inf
    
    for (k in k_range) {
      lda_try <- try(
        FitLdaModel(cluster_data, k = k, iterations = 200, burnin = 50, alpha = 0.1, beta = 0.01),
        silent = TRUE
      )
      if (inherits(lda_try, "try-error")) next
      
      # 🔧 Ensure proper format for coherence calculation
      cluster_data <- as(cluster_data, "dgCMatrix")
      
      coherence <- CalcProbCoherence(lda_try$phi, cluster_data)
      avg_coherence <- mean(coherence, na.rm = TRUE)
      
      if (!is.nan(avg_coherence) && avg_coherence > best_score) {
        best_k <- k
        best_score <- avg_coherence
      }
    }
    
    if (is.null(best_k)) {
      cat("   ❌ Failed to fit any LDA model. Skipping.\n")
      next
    }
    
    cat("   ✅ Best k =", best_k, "with coherence =", round(best_score, 4), "\n")
    
    if (best_score >= coherence_threshold) {
      cat("   ✅ Cluster is coherent. No split needed.\n")
      next
    }
    
    # 🚀 Split using best_k
    lda_model <- FitLdaModel(cluster_data, k = best_k, iterations = 500, burnin = 50)
    doc_topics <- apply(lda_model$theta, 1, which.max)
    unique_topics <- unique(doc_topics)
    
    for (topic_id in unique_topics) {
      max_cluster_id <- max_cluster_id + 1
      topic_indices <- cluster_indices[doc_topics == topic_id]
      new_clusters[topic_indices] <- max_cluster_id
    }
    
    cat("   🚀 Cluster", cluster_id, "was split into", length(unique_topics), "subclusters.\n")
  }
  
  return(new_clusters)
}


#------------------------------------------------------------------------------

# Step 1: Analyze and split clusters
# non adaptive 
#new_clusters <- split_clusters_with_lda(tfidf_reduced, kmeans_model_updated , k = 5,coherence_threshold = 0.07)

# Adaptive k - meaning k is dynamic and not fixed 
new_clusters <-split_clusters_with_lda_adaptive(
  tfidf_reduced = tfidf_reduced,        # your sparse TF-IDF matrix
  kmeans_model = kmeans_model_updated,        # your fitted K-means model
  coherence_threshold = 0.04,         # threshold for deciding when to split
  k_range = 2:10                       # range of topic counts to evaluate
)

# Step 2: Renumber clusters to ensure sequential IDs
new_clusters <- renumber_clusters(new_clusters)

# Step 2: Update K-means model
new_kmeans_model <- update_kmeans_model(kmeans_model_updated , new_clusters, tfidf_reduced)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Extract Top Terms for Each Cluster:
#------------------------------------------------------------------------------
#' 🔍 Extract Top Terms Per Cluster (Based on TF-IDF Scores)
#'
#' Given a `dgCMatrix` TF-IDF representation and a k-means model with cluster assignments,
#' this function identifies the most important terms for each cluster. Importance is
#' determined by summing TF-IDF scores across all documents within each cluster.
#'
#' 🧪 Note on Scoring:
#' This function uses **TF-IDF-weighted scores**, not raw term frequencies.
#' The TF-IDF matrix encodes both:
#'   - Term frequency in a document (TF)
#'   - Inverse document frequency across corpus (IDF)
#' The result emphasizes **highly relevant and distinctive terms per cluster.**
#'
#' @param tfidf_reduced A sparse TF-IDF matrix (`dgCMatrix`) of shape documents × terms.
#' @param kmeans_model A k-means model object with a `$cluster` vector for document assignments.
#' @param top_n Integer, number of top terms to extract per cluster. Default is 10.
#'
#' @return A named list. Each element contains a character vector of top terms for a cluster.
#' @examples
#' terms <- extract_cluster_terms(tfidf_reduced, kmeans_model, top_n = 20)
#' print(terms[["Cluster_3"]])
#' @export
extract_cluster_terms <- function(tfidf_reduced, kmeans_model, top_n = 10) {
  cat("🔹 Extracting top terms per cluster (TF-IDF weighted)...\n")
  
  cluster_terms <- list()
  unique_clusters <- sort(unique(kmeans_model$cluster))
  
  for (cluster_id in unique_clusters) {
    cat("\n📂 Processing Cluster:", cluster_id, "\n")
    
    # Subset rows (documents) for this cluster
    cluster_indices <- which(kmeans_model$cluster == cluster_id)
    cluster_data <- tfidf_reduced[cluster_indices, , drop = FALSE]
    
    # Sum TF-IDF scores across all documents in the cluster (column-wise sum)
    term_sums <- colSums(cluster_data)
    
    # Select top_n terms based on highest cumulative TF-IDF scores
    top_terms <- names(sort(term_sums, decreasing = TRUE)[1:min(top_n, length(term_sums))])
    
    # Store in list with cluster ID as name
    cluster_terms[[paste0("Cluster_", cluster_id)]] <- top_terms
    
    cat("   🔑 Top terms:", paste(top_terms, collapse = ", "), "\n")
  }
  
  cat("\n✅ Completed extracting TF-IDF-weighted top terms per cluster.\n")
  return(cluster_terms)
}

# Run the function to extract top terms
cluster_terms <- extract_cluster_terms(tfidf_reduced, new_kmeans_model, top_n = 30)

# Convert the list to a data frame
top_terms_df_lda <- data.frame(
  cluster = names(cluster_terms),
  keywords = sapply(cluster_terms, function(x) paste(x, collapse = ", "))
)

# Inspect the transformed data frame
print(top_terms_df_lda)


# Convert the dataframe content into a formatted string
df_content <- paste0(
  apply(top_terms_df_lda, 1, function(row) {
    paste0("Cluster ", row["cluster"], ": ", row["keywords"])
  }),
  collapse = "\n"
)

# Construct the full ChatGPT prompt
chatgpt_prompt <- paste0(
  "I have a dataset with clusters and associated keywords. The goal is to assign meaningful, concise labels to each cluster based on the provided keywords.\n",
  "Here is the data:\n\n",
  df_content,
  "\n\nEach cluster has a unique set of keywords that describe its theme or topic. Please generate concise and meaningful labels for each cluster, reflecting the overall subject matter."
)

# Print the prompt to verify
cat(chatgpt_prompt)
#-------------------------------------------------------------------------------
# Below is the automated label generation
#-------------------------------------------------------------------------------
#' 🧠 Generate Cluster Labels from Top Terms using ChatGPT
#'
#' This function accepts a named list of top terms per cluster (e.g., from
#' `extract_cluster_terms()`), then queries OpenAI's ChatGPT API to produce
#' short, descriptive labels for each cluster.
#'
#' TF-IDF top terms are used to summarize the content and theme of each cluster,
#' and the prompt explicitly instructs ChatGPT to infer a label based on the
#' distinctiveness of terms (TF-IDF weighted).
#'
#' @param cluster_terms A named list of clusters (e.g., "Cluster_1", "Cluster_2", ...)
#'        with character vectors of top terms.
#' @param api_key A string. Your OpenAI API key.
#' @param model The GPT model to use. Default is "gpt-4".
#' @return A named character vector of cluster labels, suitable for assignment as:
#'         `cluster_labels_Spacy <- c(...)`
#'
#' @examples
#' cluster_labels_Spacy <- generate_cluster_labels_spacy(cluster_terms, api_key = "your_api_key")
#' print(cluster_labels_Spacy["Cluster_3"])
#'
#' @export
generate_cluster_labels_spacy <- function(cluster_terms, api_key, model = "gpt-4") {
  library(httr)
  library(jsonlite)
  
  api_url <- "https://api.openai.com/v1/chat/completions"
  cluster_labels <- character(length(cluster_terms))
  names(cluster_labels) <- names(cluster_terms)
  
  for (cluster_id in names(cluster_terms)) {
    
    prompt <- paste0(
      "You are an expert in Natural Language Processing and topic modeling.\n",
      "Below is a list of top terms extracted from a document cluster using TF-IDF weighting.\n",
      "TF-IDF reflects both term frequency and distinctiveness, so the terms are highly representative of this cluster.\n\n",
      "Generate a short and meaningful label (2–5 words) that best summarizes the topic.\n",
      "Avoid simply repeating the terms unless necessary.\n\n",
      "Top terms: ", paste(cluster_terms[[cluster_id]], collapse = ", ")
    )
    
    response <- POST(
      url = api_url,
      add_headers(Authorization = paste("Bearer", api_key)),
      content_type_json(),
      body = toJSON(list(
        model = model,
        messages = list(
          list(role = "system", content = "You are a helpful assistant."),
          list(role = "user", content = prompt)
        )
      ), auto_unbox = TRUE)
    )
    
    parsed <- content(response, "parsed")
    
    label <- parsed$choices[[1]]$message$content
    cluster_labels[[cluster_id]] <- label
    
    cat("✅", cluster_id, "→", label, "\n")
  }
  
  return(cluster_labels)
}


cluster_labels_Spacy <- generate_cluster_labels_spacy(cluster_terms, api_key)

# You can now print or save:
dput(cluster_labels_Spacy)
#-------------------------------------------------------------------------------
# Here we need the cluster labels from chatgpt the following format is an example for 19 cluster label. However, this needs to change based on the number of clusters

# cluster_labels_Spacy <- c(
#Cluster_1 = "Climate Change and Environmental Policy",
#Cluster_2 = "Industrial Engineering and Technological Innovation",
#...
#)



#-------------------------------------------------------------------------------
# Here the cluster lables are updated within the model with ( )

new_kmeans_model$labels <- sapply(new_kmeans_model$cluster, function(cluster_id) {
  cluster_labels_Spacy[[paste0("Cluster_", cluster_id)]]
})

cluster_labels_Spacy <- data.frame(
  Cluster = as.numeric(sub("Cluster_", "", names(cluster_labels_Spacy))),
  Suggested_Label = unlist(cluster_labels_Spacy, use.names = FALSE),
  stringsAsFactors = FALSE
)

#-------------------------------------------------------------------------------
saveRDS(new_kmeans_model , file = "output/models/kmeans/43_new_kmeans_model_udpipe_28_03_25.rds")
# Copy to the package SectorinsightsV2 in -inst/models/kmeans/
saveRDS( tfidf_reduced , file = "output/models/dfm/43_tfidf_reduced_udpipe_28_03_25.rds")

#--------------------------------------------------------------------------------------------
# The final Structure for the Kmeans model is as follows 
#List of 10
#$ cluster        : int [1:n]         # Cluster assignment for each document
#$ centers        : num [k × d]       # Cluster centroids (k clusters × d features)
#$ totss          : num               # Total sum of squares (optional)
#$ withinss       : num [1:k]         # Within-cluster sum of squares
#$ tot.withinss   : num               # Total within-cluster sum of squares
#$ betweenss      : num               # Between-cluster sum of squares
#$ size           : int [1:k]         # Number of documents per cluster
#$ iter           : int               # Number of iterations used in kmeans
#$ ifault         : int               # Fault code (0 = success)
#$ labels         : chr [1:n]         # Custom field: topic label for each document
#- attr(*, "class"): chr "kmeans"

# 📊 Compute and Summarise Coherence Per Cluster with ChatGPT Labels
summarise_cluster_coherence <- function(tfidf_reduced, kmeans_model, top_n = 30, n_topics = 2) {
  library(textmineR)
  library(dplyr)
  library(Matrix)
  
  # Ensure matrix is in dgCMatrix format
  tfidf_reduced <- as(tfidf_reduced, "dgCMatrix")
  
  clusters <- sort(unique(kmeans_model$cluster))
  results <- list()
  
  for (clust in clusters) {
    cat("🔍 Processing Cluster:", clust, "\n")
    
    # Subset documents in this cluster
    doc_ids <- which(kmeans_model$cluster == clust)
    cluster_matrix <- tfidf_reduced[doc_ids, , drop = FALSE]
    
    # Skip small or empty clusters
    if (nrow(cluster_matrix) < 5 || sum(rowSums(cluster_matrix)) == 0) {
      cat("⚠️ Cluster", clust, "skipped (too few documents or empty).\n")
      results[[length(results)+1]] <- data.frame(
        cluster = clust,
        documents = length(doc_ids),
        coherence = NA_real_,
        label = kmeans_model$labels[doc_ids[1]]
      )
      next
    }
    
    # Fit LDA
    lda_model <- tryCatch({
      FitLdaModel(cluster_matrix, k = n_topics, iterations = 200, burnin = 50, alpha = 0.1, beta = 0.01)
    }, error = function(e) NULL)
    
    if (is.null(lda_model)) {
      cat("❌ LDA failed for Cluster", clust, "\n")
      results[[length(results)+1]] <- data.frame(
        cluster = clust,
        documents = length(doc_ids),
        coherence = NA_real_,
        label = kmeans_model$labels[doc_ids[1]]
      )
      next
    }
    
    # Calculate coherence
    coherence_vals <- CalcProbCoherence(lda_model$phi, cluster_matrix)
    avg_coherence <- mean(coherence_vals, na.rm = TRUE)
    
    cat("✅ Cluster", clust, "→ Coherence:", round(avg_coherence, 4), "\n")
    
    results[[length(results)+1]] <- data.frame(
      cluster = clust,
      documents = length(doc_ids),
      coherence = avg_coherence,
      label = kmeans_model$labels[doc_ids[1]]
    )
  }
  
  # Return summary
  summary_df <- bind_rows(results) %>% arrange(desc(coherence))
  return(summary_df)
}

coherence_summary_df <- summarise_cluster_coherence(tfidf_reduced, new_kmeans_model)
print(coherence_summary_df)
