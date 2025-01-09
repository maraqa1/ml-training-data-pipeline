#' @title Extract Topics from Description
#' @description This function sends a prompt to the OpenAI API to extract a single main topic,
#' including a domain, level 1, and possible level 2, from a given description.
#' @param prompt A character string containing the text description.
#' @param api_key A character string containing the OpenAI API key.
#' @param model A character string specifying the OpenAI model to use (default: `"gpt-3.5-turbo-instruct"`).
#' @param max_tokens The maximum number of tokens to generate (default: 150).
#' @return A cleaned character string with the extracted topic, or `NA` in case of an error.
#' @export
#' @examples
#' \dontrun{
#' openai_request_topics("Analyze the impact of renewable energy...", api_key = "your_api_key")
#' }
openai_request_topics <- function(prompt, api_key, model = "gpt-3.5-turbo-instruct", max_tokens = 150) {
  cat("\n[DEBUG] Input Prompt (Topics):\n", prompt, "\n")
  
  # Define API endpoint and headers
  url <- "https://api.openai.com/v1/completions"
  headers <- c(
    `Content-Type` = "application/json",
    `Authorization` = paste("Bearer", api_key)
  )
  
  # Construct prompt
  full_prompt <- paste(
    "Extract a single main topic, including a single domain, level 1, and possible level 2, from the following description:\n",
    prompt,
    "\nFormat output like the following example:\nDomain: Agronomics\nLevel 1: Statistical Approaches in Precision Farming\nLevel 2: High Precision Spatial Experimentation on Farm\n",
    "if Not enough information is avilable to to determine. Then note -NA-"
  )
  
  # Define request payload
  data <- list(
    model = model,
    prompt = full_prompt,
    max_tokens = max_tokens
  )
  
  # Make API request
  response <- httr::POST(
    url,
    httr::add_headers(.headers = headers),
    body = jsonlite::toJSON(data, auto_unbox = TRUE)
  )
  
  # Check response status
  if (httr::status_code(response) != 200) {
    warning("[ERROR] API request failed with status code: ", httr::status_code(response))
    return(NA)
  }
  
  # Parse response
  content <- httr::content(response, as = "parsed", simplifyVector = FALSE)
  cat("\n[DEBUG] Full API Response (Topics):\n", jsonlite::toJSON(content, pretty = TRUE), "\n")
  
  # Ensure response structure is valid
  if (!is.list(content$choices) || length(content$choices) < 1) {
    warning("[ERROR] Invalid response structure.")
    return(NA)
  }
  
  # Extract text response
  api_text <- content$choices[[1]]$text
  api_text <- gsub("\\n", " ", api_text)  # Clean newlines
  api_text <- gsub("\\s+", " ", api_text) # Clean extra spaces
  return(api_text)
}

# ------------------------------------------------------------------------------

#' @title Extract Sectors from Description
#' @description This function sends a prompt to the OpenAI API to identify the primary and 
#' secondary sectors from a given description.
#' @param prompt A character string containing the text description.
#' @param api_key A character string containing the OpenAI API key.
#' @param model A character string specifying the OpenAI model to use (default: `"gpt-3.5-turbo-instruct"`).
#' @param max_tokens The maximum number of tokens to generate (default: 150).
#' @return A list containing `Primary_Sector` and `Secondary_Sector`, or `NA` in case of an error.
#' @export
#' @examples
#' \dontrun{
#' openai_request_sectors("Analyze the role of construction in the economy...", api_key = "your_api_key")
#' }
openai_request_sectors <- function(prompt, api_key, model = "gpt-3.5-turbo-instruct", max_tokens = 150) {
  if (is.na(prompt) || prompt == "") {
    cat("[DEBUG] Skipping empty or NA prompt.\n")
    return(NA)
  }
  
  # API Endpoint and Headers
  url <- "https://api.openai.com/v1/completions"
  headers <- c(
    `Content-Type` = "application/json",
    `Authorization` = paste("Bearer", api_key)
  )
  
  # Construct the Prompt
  constructed_prompt <- paste(
    "You are an expert in topic labeling export of economy and research sectors.",
    "Your task is to identify the Primary Sector and, if applicable, the Secondary Sector from the given description.",
    
    "",
    "Provide your response in the following format:",
    "Primary_Sector: [Primary sector name]",
    "---",  # Clear separator to distinguish between primary and secondary
    "Secondary_Sector: [Secondary sector name1 (if applicable)/Secondary sector name2 (if applicable)]",
    "",
    "Example:",
    "Primary_Sector: Agriculture",
    "Secondary_Sector: Precision Farming",
    "",
    "Another example with no secondary sector:",
    "Primary_Sector: Agriculture",
    "Secondary_Sector:",
    "",
    "Another example with multi secondary sector:",
    "Primary_Sector: Agriculture",
    "Secondary_Sector: Environmental Science/Environmental Technology",
    "",
    "Please ensure that the 'Primary_Sector:' and 'Secondary_Sector:' labels are followed strictly by the sector names only, with no additional text or explanation.",
    "",
    
    "Important: Do not combine the Primary and Secondary sectors in a single line.",
    "Important: Use the exact format as shown above, with the '---' separator between the two fields.",
    "Use the exact format as shown above with the sector names on separate lines.",
    "if Not enough information is avilable to to determine. Then note -NA-",
    "",
    "Description:",
    prompt,
    sep = "\n"
  )
  
  # Debugging: Log the constructed prompt
  cat("\n[DEBUG] Constructed Prompt for Sector Request:\n", constructed_prompt, "\n")
  
  # API Payload
  data <- list(
    model = model,
    prompt = constructed_prompt,
    max_tokens = max_tokens
  )
  
  # Debugging: Log the API payload
  cat("\n[DEBUG] API Payload:\n", jsonlite::toJSON(data, pretty = TRUE), "\n")
  
  # API Request
  response <- tryCatch(
    {
      httr::POST(
        url,
        httr::add_headers(.headers = headers),
        body = jsonlite::toJSON(data, auto_unbox = TRUE)
      )
    },
    error = function(e) {
      cat("\n[ERROR] API request failed:\n", e$message, "\n")
      return(NULL) # Return NULL if the request fails
    }
  )
  
  # Validate the Response
  if (is.null(response)) {
    cat("\n[ERROR] API request returned NULL response.\n")
    return(NA)
  }
  
  # Ensure it's a valid HTTP response
  if (!inherits(response, "response")) {
    cat("\n[ERROR] Unexpected response object. Not an HTTP response.\n")
    return(NA)
  }
  
  # Check HTTP Status Code
  if (httr::status_code(response) != 200) {
    cat("\n[ERROR] API request failed. Status code:", httr::status_code(response), "\n")
    cat("[DEBUG] Response Content:\n", httr::content(response, as = "text"), "\n")
    return(NA)
  }
  
  # Parse the Response
  content <- tryCatch(
    {
      httr::content(response, "parsed", simplifyVector = TRUE)
    },
    error = function(e) {
      cat("\n[ERROR] Failed to parse API response:\n", e$message, "\n")
      return(NULL)
    }
  )
  
  # Debugging: Log the raw API response
  cat("\n[DEBUG] Full API Response (Sectors):\n", jsonlite::toJSON(content, pretty = TRUE), "\n")
  
  # Extract and clean the text response
  if (!is.null(content$choices) && length(content$choices) > 0) {
    raw_text <- content$choices[[1]]
    cat("\n[DEBUG] Raw Response Text:\n", raw_text, "\n")
    
    # Clean the response
    cleaned_text <- raw_text %>%
      gsub("\\n", " ", .) %>%  # Replace all newlines with spaces
      gsub("---", " ", .) %>% # Replace multiple spaces with a single space
      gsub("\\s+", " ", .) %>% # Replace multiple spaces with a single space
      trimws()                 # Trim leading and trailing spaces
    cat("\n[DEBUG] Cleaned Response Text:\n", cleaned_text, "\n")
    
    # Extract Primary Sector
    primary_sector <- str_extract(cleaned_text, "(?<=Primary_Sector:\\s).*?(?=\\s*Secondary_Sector:|$)")
    if (is.na(primary_sector)) {
      cat("\n[DEBUG] Failed to extract Primary_Sector from cleaned text.\n")
    } else {
      cat("\n[DEBUG] Extracted Primary_Sector:\n", primary_sector, "\n")
    }
    
    # Extract Secondary Sector
    secondary_sector <- str_extract(cleaned_text, "(?<=Secondary_Sector:\\s).*")
    if (is.na(secondary_sector)) {
      cat("\n[DEBUG] Failed to extract Secondary_Sector from cleaned text.\n")
    } else {
      cat("\n[DEBUG] Extracted Secondary_Sector:\n", secondary_sector, "\n")
    }
    
    # Return both sectors as a named list for further processing
    return(list(
      Primary_Sector = primary_sector,
      Secondary_Sector = secondary_sector
    ))
  } else {
    cat("\n[ERROR] Unexpected API response format.\n")
    return(NA)
  }
}
#-------------------------------------------------------------------------------
#' @title Process a Dataframe with OpenAI APIs
#' @description This function processes a dataframe by extracting topics and sector classifications 
#' for each row. It uses the `openai_request_topics` and `openai_request_sectors` functions for API calls.
#' @param data A dataframe containing a `PublicDescription` column.
#' @param api_key A character string containing your OpenAI API key.
#' @param model A character string specifying the OpenAI model to use. Defaults to `"gpt-3.5-turbo-instruct"`.
#' @return A dataframe with the following added columns:
#' \itemize{
#'   \item `topic_response`: Raw topic response from the OpenAI API.
#'   \item `Domain`: Extracted domain from the topic response.
#'   \item `Level_1`: Extracted level 1 topic detail.
#'   \item `Level_2`: Extracted level 2 topic detail.
#'   \item `combined_features`: Combined textual description of all extracted topic fields.
#'   \item `Primary_Sector`: Extracted primary sector from the sector response.
#'   \item `Secondary_Sector`: Extracted secondary sector from the sector response.
#' }
#' @export
#' @examples
#' \dontrun{
#' test_data <- tibble(
#'   PublicDescription = c(
#'     "Analyze the economic impact of renewable energy solutions...",
#'     "Developing AI-driven tools for sentiment analysis in business applications..."
#'   )
#' )
process_dataframe_with_sectors <- function(data, api_key, model = "gpt-3.5-turbo-instruct") {
  # Ensure the input data has the required column
  if (!"PublicDescription" %in% colnames(data)) {
    stop("[ERROR] Input dataframe must contain a 'PublicDescription' column.")
  }
  
  # Process each row of the dataframe
  processed_data <- data %>%
    mutate(
      # First API Request: Topics
      topic_response = map_chr(PublicDescription, ~ {
        tryCatch(openai_request_topics(.x, api_key, model), error = function(e) {
          cat("\n[ERROR] Topics request failed for description:\n", .x, "\n", e$message, "\n")
          return(NA)
        })
      }),
      
      # Extract Fields from Topic Response
      Domain = map_chr(topic_response, ~ ifelse(!is.na(.x), str_extract(.x, "(?<=Domain: ).*?(?= Level 1:)"), NA)),
      Level_1 = map_chr(topic_response, ~ ifelse(!is.na(.x), str_extract(.x, "(?<=Level 1: ).*?(?= Level 2:)"), NA)),
      Level_2 = map_chr(topic_response, ~ ifelse(!is.na(.x), str_extract(.x, "(?<=Level 2: ).*"), NA)),
      combined_features = pmap_chr(list(Domain, Level_1, Level_2), ~ paste(c(...), collapse = " ") %>% trimws()),
      
      # Second API Request: Sectors
      sector_response = map(PublicDescription, ~ {
        tryCatch(openai_request_sectors(.x, api_key, model), error = function(e) {
          cat("\n[ERROR] Sectors request failed for description:\n", .x, "\n", e$message, "\n")
          return(list(Primary_Sector = NA, Secondary_Sector = NA))
        })
      }),
      
      # Extract Fields from Sector Response
      Primary_Sector = map_chr(sector_response, ~ .x$Primary_Sector),
      Secondary_Sector = map_chr(sector_response, ~ .x$Secondary_Sector)
    )
  
  return(processed_data)
}

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

api_key <- "sk-proj-7xmG19eewuG8cUM89TaPT3BlbkFJdXxzFQ2yzuirj96y6FK1"

# Load Libraries
library(httr)
library(tm)
library(tidytext)
library(tidyverse)
library(furrr)
library(future)
library(progressr)

# Set up parallel processing with all available cores
plan(multisession, workers = availableCores())

# API Key
# api_key <- "your_api_key_here"

# Input and Output Paths
input_file <- "input/updated_file1.csv"  # Input file
output_dir <- "output/"           # Output directory for chunks and final results

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

#-------------------------------------------------------------------------------

#' @title Split Data into Chunks
#' @description Splits a dataframe into smaller chunks for batch processing.
#' @param data A dataframe to be split.
#' @param chunk_size An integer specifying the number of rows per chunk. Defaults to 1000.
#' @return A list of dataframes, each containing up to `chunk_size` rows.
#' @export
#' @examples
#' \dontrun{
#' chunks <- chunk_data(data, chunk_size = 500)
#' }
chunk_data <- function(data, chunk_size = 1000) {
  num_chunks <- ceiling(nrow(data) / chunk_size)
  split(data, rep(1:num_chunks, each = chunk_size, length.out = nrow(data)))
}
#-------------------------------------------------------------------------------

#' @title Process a Data Chunk
#' @description Processes a chunk of data by extracting topics and sectors for each row using the OpenAI API.
#' @param chunk A dataframe chunk containing a `PublicDescription` column.
#' @param api_key A character string containing your OpenAI API key.
#' @return A processed dataframe chunk with extracted fields.
#' @export
#' @examples
#' \dontrun{
#' processed_chunk <- process_chunk(data_chunk, api_key = "your_api_key")
#' }
process_chunk <- function(chunk, api_key) {
  with_progress({
    p <- progressor(steps = nrow(chunk))
    
    # Process the chunk
    processed_chunk <- chunk %>%
      mutate(
        # First API Request: Topics
        topic_response = map_chr(PublicDescription, ~ {
          tryCatch(openai_request_topics(.x, api_key), error = function(e) {
            cat("\n[ERROR] Topics request failed for description:\n", .x, "\n", e$message, "\n")
            return(NA)
          })
        }),
        Domain = map_chr(topic_response, ~ ifelse(!is.na(.x), str_extract(.x, "(?<=Domain: ).*?(?= Level 1:)"), NA)),
        Level_1 = map_chr(topic_response, ~ ifelse(!is.na(.x), str_extract(.x, "(?<=Level 1: ).*?(?= Level 2:)"), NA)),
        Level_2 = map_chr(topic_response, ~ ifelse(!is.na(.x), str_extract(.x, "(?<=Level 2: ).*"), NA)),
        combined_features = pmap_chr(list(Domain, Level_1, Level_2), ~ paste(c(...), collapse = " ") %>% trimws()),
        
        # Second API Request: Sectors
        sector_response = map(PublicDescription, ~ {
          tryCatch(openai_request_sectors(.x, api_key), error = function(e) {
            cat("\n[ERROR] Sectors request failed for description:\n", .x, "\n", e$message, "\n")
            return(list(Primary_Sector = NA, Secondary_Sector = NA))
          })
        }),
        
        # Safely extract Primary_Sector and Secondary_Sector
        Primary_Sector = map_chr(sector_response, ~ {
          if (is.list(.x)) {
            return(ifelse(is.null(.x$Primary_Sector), NA, .x$Primary_Sector))
          } else if (is.character(.x)) {
            # If sector_response is a string (unexpected case), try to extract Primary_Sector
            return(str_extract(.x, "(?<=Primary_Sector:).*?(?=\\s+Secondary_Sector:)") %>% trimws())
          } else {
            return(NA)
          }
        }),
        
        Secondary_Sector = map_chr(sector_response, ~ {
          if (is.list(.x)) {
            return(ifelse(is.null(.x$Secondary_Sector), NA, .x$Secondary_Sector))
          } else if (is.character(.x)) {
            # If sector_response is a string (unexpected case), try to extract Secondary_Sector
            return(str_extract(.x, "(?<=Secondary_Sector:).*") %>% trimws())
          } else {
            return(NA)
          }
        })
      ) %>%
      select(-sector_response) # Drop the sector_response column
    
    p()  # Update progress bar for each processed row
    return(processed_chunk)
  })
}
#-------------------------------------------------------------------------------

#' @title Consolidate Processed Chunks
#' @description Consolidates all processed data chunks into a single dataframe and saves it to a CSV file.
#' @param output_dir A character string specifying the directory containing chunk files.
#' @param consolidated_file A character string specifying the path to the consolidated CSV file.
#' @return None. The consolidated results are saved to the specified file.
#' @export
#' @examples
#' \dontrun{
#' consolidate_results(output_dir = "path/to/output", consolidated_file = "final_results.csv")
#' }
#' 
consolidate_results <- function(output_dir, consolidated_file) {
  # List all chunk files
  chunk_files <- list.files(output_dir, pattern = "chunk_.*\\.csv", full.names = TRUE)
  
  # Read all chunk files into a list of data frames
  all_chunks <- lapply(chunk_files, read.csv)
  
  # Bind all data frames into a single data frame
  final_df <- bind_rows(all_chunks)
  
  # Write the consolidated results to a CSV file
  write.csv(final_df, consolidated_file, row.names = FALSE)
  cat("Consolidated results saved to:", consolidated_file, "\n")
}

#' @title Process a Large File with OpenAI API
#' @description Processes a large input file by chunking the data, sending API requests to extract topics 
#' and sectors, and consolidating the results into a final CSV file.
#' @param input_file A character string specifying the path to the input CSV file.
#' @param output_dir A character string specifying the directory to save intermediate chunk files.
#' @param api_key A character string containing your OpenAI API key.
#' @param sample_size An integer specifying the number of rows to sample from the input file. 
#' Set to `NULL` to process the full dataset.
#' @param chunk_size An integer specifying the number of rows per chunk. Defaults to 1000.
#' @return None. The final consolidated results are saved to a file in the output directory.
#' @export
#' @examples
#' \dontrun{
#' process_large_file(
#'   input_file = "path/to/input.csv",
#'   output_dir = "path/to/output/",
#'   api_key = "your_api_key",
#'   sample_size = 100,
#'   chunk_size = 10
#' )
#' }

# Main Workflow
process_large_file <- function(input_file, output_dir, api_key, sample_size = NULL, chunk_size = 1000) {
  # Step 1: Read Input Data
  input_data <- read.csv(input_file, stringsAsFactors = FALSE) %>%
    select(doc_id, PublicDescription)
  
  if (nrow(input_data) == 0) stop("[ERROR] Input file is empty or invalid.")
  
  # Step 2: Take a Sample (Optional)
  if (!is.null(sample_size)) {
    input_data <- sample_n(input_data, sample_size)
  }
  
  # Step 3: Chunk Data
  cat("[INFO] Splitting data into chunks...\n")
  data_chunks <- chunk_data(input_data, chunk_size)
  
  # Step 4: Process Each Chunk in Parallel
  cat("[INFO] Processing chunks in parallel...\n")
  output_files <- future_map(seq_along(data_chunks), function(chunk_idx) {
    chunk <- data_chunks[[chunk_idx]]
    cat("[INFO] Processing chunk", chunk_idx, "with", nrow(chunk), "rows.\n")
    
    # Process the chunk
    processed_chunk <- process_chunk(chunk, api_key)
    
    # Save the processed chunk to a file
    chunk_file <- file.path(output_dir, paste0("chunk_", chunk_idx, ".csv"))
    write.csv(processed_chunk, chunk_file, row.names = FALSE)
    cat("[INFO] Processed chunk", chunk_idx, "saved to", chunk_file, "\n")
    return(chunk_file)
  }, .progress = TRUE)
  
  # Step 5: Consolidate Results
  consolidated_file <- file.path(output_dir, "consolidated_results.csv")
  consolidate_results(output_dir, consolidated_file)
}

#-------------------------------------------------------------------------------
# Execute the Workflow
process_large_file(
  input_file = input_file,
  output_dir = output_dir,
  api_key = api_key,
  sample_size = 100,      # Set to NULL to process the full dataset
  chunk_size = 10         # Adjust chunk size based on system capacity
)


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#' @title Test Workflow with Example Data
#' @description A sample workflow to test the entire pipeline on example data.
#' @return None. Prints the processed results.
#' @examples
#' \dontrun{
#' result <- process_dataframe_with_sectors(test_data, api_key = "your_api_key")
#' print(result)
#' }
test_data <- tibble(
  PublicDescription = c(
    "More challenging environmental standards (Code 5+) add considerable complexity to housing design...",
    "Exploring renewable energy solutions for rural communities with a focus on solar panel integration...",
    "Developing AI-driven tools for multilingual sentiment analysis in business applications..."
  )
)
#-------------------------------------------------------------------------------

# Replace with your OpenAI API key
#api_key <- "your_api_key_here"




# Process the dataframe
result_df <- process_dataframe_with_sectors(data = test_data, api_key = api_key)

# Print the results
print(result_df)


results_df<-read.csv("output/consolidated_results.csv")
View(results_df)

