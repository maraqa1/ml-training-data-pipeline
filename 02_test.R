# Load necessary libraries
library(tidyverse)
library(tm)
library(textclean)
library(tidytext)
library(text)
library(quanteda)
library(cluster)
library(udpipe)
library(furrr)
library(progressr)

# Step 1: Load Data
data <- read_csv("output/consolidated_results.csv")

# Combine features into a single text field for each document
data <- data %>%
  select(-combined_features) %>%
  mutate(combined_features = paste(Domain, Level_1, Level_2, Primary_Sector, Secondary_Sector, sep = " "))

# Step 2: Text Cleaning Function
clean_text <- function(text) {
  text %>%
    stringr::str_to_lower() %>%
    textclean::replace_contraction() %>%
    tm::removePunctuation() %>%
    tm::removeNumbers() %>%
    tm::stripWhitespace() %>%
    tm::removeWords(c(tm::stopwords("en"), "project", "focuses", "team", "users", 
                      "excuse", "tongue", "sticking", "loring"))
}

# Clean combined features
data <- data %>%
  mutate(cleaned_text = map_chr(combined_features, clean_text))

# Step 3: Download and Load UDPipe Model
model_path <- "english-ewt-ud-2.5-191206.udpipe"
if (!file.exists(model_path)) {
  udpipe_download_model(language = "english", model_dir = dirname(model_path))
}

# Step 4: Define a Parallel-Safe Function
extract_cleaned_text_parallel <- function(description, model_path) {
  ud_model <- udpipe_load_model(file = model_path)  # Load UDPipe model in worker
  
  clean_text <- str_replace_all(description, "[[:punct:]]", "") %>% tolower()
  annotations <- udpipe_annotate(ud_model, x = clean_text)
  annotations <- as.data.frame(annotations)
  
  keywords <- annotations %>%
    filter(upos %in% c("NOUN", "VERB")) %>%
    pull(token) %>%
    unique() %>%
    paste(collapse = ", ")
  
  return(keywords)
}


# Step 5: Apply Parallel Processing
plan(multisession, workers = 1)  # Set up parallel processing

# Define the parallel-safe extraction function
extract_cleaned_text_parallel <- function(description, model_path) {
  # Reload the UDPipe model inside the worker
  ud_model <- udpipe_load_model(file = model_path)
  
  # Preprocess the text
  clean_text <- str_replace_all(description, "[[:punct:]]", "") %>% tolower()
  
  # Annotate the text using UDPipe
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

# Process data in parallel
library(progressr)  # For progress bar
with_progress({
  p <- progressor(along = data$cleaned_text)  # Initialize progress bar
  
  # Use future_map_chr to process in parallel
  data <- data %>%
    mutate(cleaned_text_1 = future_map_chr(
      cleaned_text,
      ~ {
        p()  # Update progress bar
        extract_cleaned_text_parallel(.x, model_path)  # Call the parallel-safe function
      },
      .options = furrr_options(seed = 123)  # Set a fixed seed for reproducibility
    ))
})

# Step 6: Return to Sequential Processing
plan(sequential)  # Return to sequential processing after parallel operations

# Save the processed data
write_csv(data, "output/processed_data_with_keywords.csv")
