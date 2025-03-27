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

# Step 3: Replace spaCy with udpipe for extracting nouns and verbs
# Load the English model
ud_model <- udpipe_download_model(language = "english")
ud_model <- udpipe_load_model(file = ud_model$file_model)

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



# Apply the function to extract cleaned text
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


# Step 4 onwards remains unchanged...
# (Continue as in the original script)

# Summary and clustering steps will remain the same as they do not depend on spaCy.
data %>% dplyr::select (PublicDescription,cleaned_text_1) %>% head()

write.csv(data,"output/features.csv")
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
write_csv(chatgpt_prompt, "output/chatgpt_prompt.csv")

#-------------------------------------------------------------------------------
# TUNE the Clusters using LDA

#-------------------------------------------------------------------------------

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


# Step 1: Analyze and split clusters
new_clusters <- split_clusters_with_lda(tfidf_reduced, kmeans_model_updated , k = 5,coherence_threshold = 0.07)

# Step 2: Renumber clusters to ensure sequential IDs
new_clusters <- renumber_clusters(new_clusters)

# Step 2: Update K-means model
new_kmeans_model <- update_kmeans_model(kmeans_model_updated , new_clusters, tfidf_reduced)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Extract Top Terms for Each Cluster:
extract_cluster_terms <- function(tfidf_reduced, kmeans_model, top_n = 10) {
  cat("Extracting top terms per cluster...\n")
  
  # Initialize a list to store terms per cluster
  cluster_terms <- list()
  
  # Get unique clusters
  unique_clusters <- unique(kmeans_model$cluster)
  
  # Iterate over each cluster
  for (cluster_id in unique_clusters) {
    cat("\nProcessing Cluster:", cluster_id, "\n")
    
    # Subset data for the current cluster
    cluster_indices <- which(kmeans_model$cluster == cluster_id)
    cluster_data <- tfidf_reduced[cluster_indices, , drop = FALSE]
    
    # Sum term frequencies across all documents in the cluster
    term_sums <- colSums(cluster_data)
    
    # Get the top N terms for the cluster
    top_terms <- names(sort(term_sums, decreasing = TRUE)[1:top_n])
    cluster_terms[[paste0("Cluster_", cluster_id)]] <- top_terms
    
    cat("   Top terms for Cluster", cluster_id, ":", paste(top_terms, collapse = ", "), "\n")
  }
  
  cat("\nCompleted extracting terms for all clusters.\n")
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
# Here we need the cluster labels from chatgpt the following format is an example for 19 cluster label. However, this needs to change based on the number of clusters

# cluster_labels_Spacy <- c(
#Cluster_1 = "Climate Change and Environmental Policy",
#Cluster_2 = "Industrial Engineering and Technological Innovation",
#...
#)


cluster_labels_Spacy <- c(
  Cluster_1 = "Climate Change and Environmental Policy",
  Cluster_2 = "Industrial Engineering and Technological Innovation",
  Cluster_3 = "Water Resource Management and Pollution Control",
  Cluster_4 = "Healthcare, Diagnostics, and Biotechnology",
  Cluster_5 = "Electric and Sustainable Transportation",
  Cluster_6 = "Construction, Architecture, and Building Technologies",
  Cluster_7 = "Aerospace and Defense Engineering",
  Cluster_8 = "Artificial Intelligence and Data Science",
  Cluster_9 = "Supply Chain and Logistics Management",
  Cluster_10 = "Fleet Electrification and Telematics Solutions",
  Cluster_11 = "Agriculture, Food Systems, and AgriTech",
  Cluster_12 = "Emissions Monitoring and Energy Extraction",
  Cluster_13 = "Hydrogen Fuel Cells and Biotech Energy Systems",
  Cluster_14 = "Wound Care and Biomedical Devices",
  Cluster_15 = "Immersive Media, Entertainment, and Creative Technologies",
  Cluster_16 = "Education Technologies and Learning Strategies",
  Cluster_17 = "Business Strategy, Analytics, and Market Insights",
  Cluster_18 = "Marketing, Customer Engagement, and Digital Services",
  Cluster_19 = "Cybersecurity and Information Protection"
)



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
saveRDS(new_kmeans_model , file = "output/models/dfm/19_new_kmeans_model_udpipe_23_03_25.rds")
# Copy to the package SectorinsightsV2 in -inst/models/kmeans/
saveRDS( tfidf_reduced , file = "output/models/kmeans/19_tfidf_reduced_udpipe_23_03_25.rds")

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




