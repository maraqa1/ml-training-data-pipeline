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
#data <- read_csv("C:/R_Home/UK_Innovate_topic_modelling/dirty_work/consolidated_document_with_sector_1st_complete_02_09_2024.csv")

data <- read_csv("output/consolidated_results.csv")

# Combine features into a single text field for each document
data <- data %>%
  select(-combined_features) %>%
  mutate(combined_features = paste(Domain, Level_1, Level_2,Primary_Sector, Secondary_Sector, sep = " "))



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

# Set up parallel backend
plan(multisession)  # Use multiple R sessions for parallel processing

# Use parallel processing for mutate
data <- data %>%
  mutate(cleaned_text_1 = future_map_chr(cleaned_text, ~ extract_cleaned_text(.x)))

# Shut down parallel workers
plan(sequential)  # Return to sequential processing

# Step 4 onwards remains unchanged...
# (Continue as in the original script)

# Summary and clustering steps will remain the same as they do not depend on spaCy.
data %>% dplyr::select (cleaned_text_1) %>% head()



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

tfidf_reduced <- reduce_dfm(tfidf_features, top_n = 1000)

# Step 6: Clustering to Group Similar Themes
apply_clustering <- function(features, k = 15) {  # Adjust `k` based on dataset size
  set.seed(123)
  kmeans_model <- stats::kmeans(as.matrix(features), centers = k)
  return(kmeans_model)
}

# Apply clustering
kmeans_model_updated <- apply_clustering(tfidf_reduced , k = 15)


# Step 7: Extract Top Terms from K-means Model
extract_top_terms_from_kmeans <- function(kmeans_model, tfidf_matrix, top_n = 10) {
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

