# ==========================================================
# Fake Seller & Review Detection App - FIXED FOR OLD IGRAPH
# ==========================================================

library(shiny)
library(tidyverse)
library(randomForest)
library(xgboost)
library(caret)
library(lubridate)
library(gridExtra)
library(DT)
library(igraph)

# ===================== HELPER FUNCTIONS =====================
load_seller_data <- function(file_input) {
  if (!is.null(file_input)) {
    tryCatch({
      data <- read.csv(file_input$datapath)
      required_cols <- c("seller_id", "account_age", "listing_freq", 
                         "avg_review_score", "num_products")
      
      if (!all(required_cols %in% colnames(data))) {
        stop("Missing required columns: ", 
             paste(setdiff(required_cols, colnames(data)), collapse = ", "))
      }
      
      if (!"fraud_label" %in% colnames(data)) {
        data$fraud_label <- 0
      }
      
      return(data)
    }, error = function(e) {
      stop("Seller file error: ", e$message)
    })
  } else {
    set.seed(123)
    n_sellers <- 200
    data <- data.frame(
      seller_id = paste0("S", 1:n_sellers),
      account_age = sample(1:100, n_sellers, TRUE),
      listing_freq = rpois(n_sellers, 10),
      avg_review_score = round(runif(n_sellers, 1, 5), 1),
      num_products = rpois(n_sellers, 50)
    )
    data$fraud_label <- ifelse(
      data$account_age < 50 & data$listing_freq > 5 & 
        data$avg_review_score > 3.8, 1, 0
    )
    
    fraud_sellers <- sample(data$seller_id, 10)
    data$fraud_label[data$seller_id %in% fraud_sellers] <- 1
    
    return(data)
  }
}

load_review_data <- function(file_input, seller_ids) {
  if (!is.null(file_input)) {
    tryCatch({
      data <- read.csv(file_input$datapath)
      required_cols <- c("review_id", "seller_id", "rating", "review_time")
      
      if (!all(required_cols %in% colnames(data))) {
        stop("Missing required columns: ", 
             paste(setdiff(required_cols, colnames(data)), collapse = ", "))
      }
      
      data$review_time <- parse_date_time(
        data$review_time, 
        orders = c("ymd HMS", "mdy HMS", "dmy HMS", "ymd", "mdy", "dmy")
      )
      
      if (!"fraud_label" %in% colnames(data)) {
        data$fraud_label <- 0
      }
      
      return(data)
    }, error = function(e) {
      stop("Review file error: ", e$message)
    })
  } else {
    n_reviews <- 1000
    
    fraud_ring_1 <- sample(seller_ids, 5)
    fraud_ring_2 <- sample(setdiff(seller_ids, fraud_ring_1), 5)
    
    fake_reviews <- c(
      "Amazing product highly recommend",
      "Best purchase ever 5 stars",
      "Excellent quality fast shipping",
      "Perfect item as described",
      "Great seller will buy again"
    )
    
    data <- data.frame(
      review_id = paste0("R", 1:n_reviews),
      seller_id = sample(seller_ids, n_reviews, TRUE),
      rating = sample(1:5, n_reviews, TRUE, prob = c(0.1, 0.15, 0.2, 0.25, 0.3)),
      review_time = sample(seq(
        as.POSIXct("2024-01-01"),
        as.POSIXct("2024-12-31"),
        by = "hour"), n_reviews, TRUE),
      review_text = sample(c(
        "Great product", "Loved it", "Terrible",
        "Fake item", "Excellent quality", "Worst ever"
      ), n_reviews, TRUE),
      emb1 = rnorm(n_reviews),
      emb2 = rnorm(n_reviews),
      fraud_label = 0
    )
    
    ring1_reviews <- 80
    for (i in 1:ring1_reviews) {
      idx <- nrow(data) + 1
      data[idx, ] <- data.frame(
        review_id = paste0("R", idx),
        seller_id = sample(fraud_ring_1, 1),
        rating = 5,
        review_time = as.POSIXct("2024-06-15") + sample(1:1000, 1) * 3600,
        review_text = sample(fake_reviews, 1),
        emb1 = rnorm(1, mean = 2),
        emb2 = rnorm(1, mean = 2),
        fraud_label = 1
      )
    }
    
    ring2_reviews <- 60
    for (i in 1:ring2_reviews) {
      idx <- nrow(data) + 1
      data[idx, ] <- data.frame(
        review_id = paste0("R", idx),
        seller_id = sample(fraud_ring_2, 1),
        rating = 5,
        review_time = as.POSIXct("2024-08-20") + sample(1:800, 1) * 3600,
        review_text = sample(fake_reviews, 1),
        emb1 = rnorm(1, mean = -2),
        emb2 = rnorm(1, mean = 2),
        fraud_label = 1
      )
    }
    
    return(data)
  }
}

# Custom BFS/DFS implementation that works with any igraph version
custom_bfs <- function(graph, start) {
  n <- vcount(graph)
  visited <- rep(FALSE, n)
  order <- numeric(0)
  queue <- start
  
  node_idx <- which(V(graph)$name == start)
  visited[node_idx] <- TRUE
  
  while (length(queue) > 0) {
    current <- queue[1]
    queue <- queue[-1]
    order <- c(order, which(V(graph)$name == current))
    
    current_idx <- which(V(graph)$name == current)
    neighbors <- neighbors(graph, current_idx)
    
    for (neighbor in neighbors) {
      if (!visited[neighbor]) {
        visited[neighbor] <- TRUE
        queue <- c(queue, V(graph)$name[neighbor])
      }
    }
  }
  
  return(list(order = order))
}

custom_dfs <- function(graph, start) {
  n <- vcount(graph)
  visited <- rep(FALSE, n)
  order <- numeric(0)
  stack <- start
  
  while (length(stack) > 0) {
    current <- stack[length(stack)]
    stack <- stack[-length(stack)]
    
    current_idx <- which(V(graph)$name == current)
    
    if (!visited[current_idx]) {
      visited[current_idx] <- TRUE
      order <- c(order, current_idx)
      
      neighbors <- neighbors(graph, current_idx)
      for (neighbor in rev(neighbors)) {
        if (!visited[neighbor]) {
          stack <- c(stack, V(graph)$name[neighbor])
        }
      }
    }
  }
  
  return(list(order = order))
}

# ===================== UI =====================
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      
      body {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
      }
      
      .main-container {
        background: white;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        margin: 30px auto;
        max-width: 1400px;
        overflow: hidden;
      }
      
      .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px 40px;
        color: white;
      }
      
      .app-header h1 {
        margin: 0;
        font-size: 32px;
        font-weight: 700;
      }
      
      .app-header .subtitle {
        margin: 8px 0 0 0;
        opacity: 0.9;
        font-size: 16px;
      }
      
      .sidebar {
        background: #f8f9fa;
        padding: 30px 25px;
        border-right: 1px solid #e9ecef;
      }
      
      .sidebar h4 {
        color: #495057;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 15px;
      }
      
      .btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        padding: 12px 30px;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        font-size: 16px;
      }
      
      .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
      }
      
      .nav-tabs {
        border-bottom: 2px solid #e9ecef;
        padding: 0 30px;
        background: white;
      }
      
      .nav-tabs .nav-link {
        border: none;
        color: #6c757d;
        font-weight: 500;
        padding: 15px 25px;
      }
      
      .nav-tabs .nav-link.active {
        color: #667eea;
        border: none;
        border-bottom: 3px solid #667eea;
        background: transparent;
      }
      
      .info-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 15px 20px;
        border-radius: 8px;
        margin-top: 20px;
      }
      
      .info-box p {
        margin: 5px 0;
        font-size: 13px;
        color: #6c757d;
      }
      
      .plot-container {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 25px;
      }
      
      pre {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 20px;
        font-size: 13px;
      }
    "))
  ),
  
  div(class = "main-container",
      div(class = "app-header",
          h1("🔍 Fraud Detection Dashboard"),
          p(class = "subtitle", "ML-Powered Seller & Review Analysis with Graph Network Detection")
      ),
      
      sidebarLayout(
        sidebarPanel(
          class = "sidebar",
          width = 3,
          
          h4("Data Upload"),
          fileInput("seller_file", "Seller Data (CSV)", accept = ".csv"),
          fileInput("review_file", "Review Data (CSV)", accept = ".csv"),
          
          hr(),
          
          h4("Analysis Settings"),
          selectInput("graph_algo", "Graph Traversal",
                      choices = c("BFS" = "BFS", "DFS" = "DFS")),
          sliderInput("train_split", "Training Split (%)", 
                      min = 60, max = 90, value = 80, step = 5),
          
          actionButton("run_analysis", "Run Detection", class = "btn-primary"),
          
          div(class = "info-box",
              p("💡 Leave uploads blank for sample data"),
              p("📊 Adjust split for validation")
          )
        ),
        
        mainPanel(
          width = 9,
          tabsetPanel(
            tabPanel("Seller Analysis",
                     div(class = "plot-container",
                         plotOutput("seller_plot", height = "400px")),
                     verbatimTextOutput("seller_metrics"),
                     DTOutput("seller_table")),
            
            tabPanel("Review Analysis",
                     div(class = "plot-container",
                         plotOutput("review_plot", height = "400px")),
                     verbatimTextOutput("review_metrics"),
                     DTOutput("review_table")),
            
            tabPanel("Network Analysis",
                     div(class = "plot-container",
                         plotOutput("graph_plot", height = "600px")),
                     verbatimTextOutput("graph_text"))
          )
        )
      )
  )
)

# ===================== SERVER =====================
server <- function(input, output, session) {
  
  observeEvent(input$run_analysis, {
    
    withProgress(message = 'Running Analysis...', value = 0, {
      
      incProgress(0.1, detail = "Loading data")
      
      tryCatch({
        seller_data <- load_seller_data(input$seller_file)
        review_data <- load_review_data(input$review_file, seller_data$seller_id)
        
        if (length(unique(seller_data$fraud_label)) < 2) {
          seller_data$fraud_label[sample(seq_len(nrow(seller_data)), 1)] <- 1
        }
        if (sum(review_data$fraud_label) == 0) {
          review_data$fraud_label[sample(seq_len(nrow(review_data)), 1)] <- 1
        }
        
      }, error = function(e) {
        showNotification(paste("Error:", e$message), type = "error", duration = 10)
        return()
      })
      
      incProgress(0.2, detail = "Training seller model")
      
      set.seed(42)
      train_idx <- createDataPartition(seller_data$fraud_label, 
                                       p = input$train_split/100, 
                                       list = FALSE)
      
      seller_train <- seller_data[train_idx, ]
      seller_test <- seller_data[-train_idx, ]
      
      seller_features_train <- seller_train %>%
        select(account_age, listing_freq, avg_review_score, num_products)
      
      seller_features_test <- seller_test %>%
        select(account_age, listing_freq, avg_review_score, num_products)
      
      rf_model <- randomForest(
        x = seller_features_train,
        y = as.factor(seller_train$fraud_label),
        ntree = 200,
        importance = TRUE
      )
      
      seller_features_all <- seller_data %>%
        select(account_age, listing_freq, avg_review_score, num_products)
      
      seller_data$fraud_score <- predict(
        rf_model, seller_features_all, type = "prob"
      )[,2]
      
      test_pred <- predict(rf_model, seller_features_test, type = "prob")[,2]
      test_pred_class <- ifelse(test_pred > 0.5, 1, 0)
      seller_cm <- confusionMatrix(
        as.factor(test_pred_class),
        as.factor(seller_test$fraud_label)
      )
      
      incProgress(0.3, detail = "Training review model")
      
      review_train <- review_data[train_idx[train_idx <= nrow(review_data)], ]
      review_test <- review_data[-train_idx[train_idx <= nrow(review_data)], ]
      
      review_features_train <- review_train %>%
        mutate(hour = hour(review_time)) %>%
        select(rating, emb1, emb2, hour)
      
      review_features_test <- review_test %>%
        mutate(hour = hour(review_time)) %>%
        select(rating, emb1, emb2, hour)
      
      dtrain <- xgb.DMatrix(
        data = as.matrix(review_features_train),
        label = review_train$fraud_label
      )
      
      dtest <- xgb.DMatrix(
        data = as.matrix(review_features_test),
        label = review_test$fraud_label
      )
      
      xgb_model <- xgb.train(
        params = list(
          objective = "binary:logistic",
          eval_metric = "auc"
        ),
        data = dtrain,
        nrounds = 50,
        verbose = 0,
        evals = list(train = dtrain, test = dtest)
      )
      
      review_features_all <- review_data %>%
        mutate(hour = hour(review_time)) %>%
        select(rating, emb1, emb2, hour)
      
      review_data$fraud_score <- predict(
        xgb_model, as.matrix(review_features_all)
      )
      
      test_pred_review <- predict(xgb_model, dtest)
      test_pred_class_review <- ifelse(test_pred_review > 0.5, 1, 0)
      review_cm <- confusionMatrix(
        as.factor(test_pred_class_review),
        as.factor(review_test$fraud_label)
      )
      
      incProgress(0.4, detail = "Building seller network")
      
      suspicious_reviews <- review_data %>%
        filter(fraud_score > 0.5) %>%
        select(seller_id, review_id)
      
      if (nrow(suspicious_reviews) < 10) {
        suspicious_reviews <- review_data %>%
          filter(fraud_score > 0.3) %>%
          select(seller_id, review_id)
      }
      
      if (nrow(suspicious_reviews) < 10) {
        suspicious_reviews <- review_data %>%
          filter(fraud_label == 1) %>%
          select(seller_id, review_id)
      }
      
      if (nrow(suspicious_reviews) < 2) {
        showNotification(
          paste("Not enough suspicious reviews found."), 
          type = "warning", duration = 10)
        return()
      }
      
      seller_connections <- review_data %>%
        filter(review_id %in% suspicious_reviews$review_id) %>%
        select(seller_id, review_text) %>%
        group_by(review_text) %>%
        filter(n() >= 2) %>%
        ungroup()
      
      seller_pairs <- seller_connections %>%
        inner_join(seller_connections, by = "review_text", 
                   relationship = "many-to-many") %>%
        filter(seller_id.x < seller_id.y) %>%
        select(from = seller_id.x, to = seller_id.y, review_text) %>%
        group_by(from, to) %>%
        summarise(shared_reviews = n(), .groups = "drop")
      
      if (nrow(seller_pairs) == 0) {
        showNotification("No seller connections found.", type = "warning", duration = 10)
        return()
      }
      
      g <- graph_from_data_frame(seller_pairs, directed = FALSE)
      
      for (v in V(g)$name) {
        seller_idx <- which(seller_data$seller_id == v)
        if (length(seller_idx) > 0) {
          V(g)[v]$fraud_score <- seller_data$fraud_score[seller_idx]
          V(g)[v]$account_age <- seller_data$account_age[seller_idx]
        }
      }
      
      incProgress(0.5, detail = "Running graph traversal")
      
      comps <- components(g)
      largest_comp <- which.max(comps$csize)
      comp_nodes <- V(g)$name[comps$membership == largest_comp]
      
      if (length(comp_nodes) < 2) {
        showNotification("Network too small.", type = "warning", duration = 10)
        return()
      }
      
      sub_g <- induced_subgraph(g, comp_nodes)
      
      fraud_scores <- sapply(V(sub_g)$name, function(v) V(sub_g)[v]$fraud_score)
      start_node <- V(sub_g)$name[which.max(fraud_scores)]
      
      # Use custom BFS/DFS that works with all igraph versions
      if (input$graph_algo == "BFS") {
        traversal_result <- custom_bfs(sub_g, start_node)
        visited_nodes <- V(sub_g)$name[traversal_result$order]
        node_distances <- distances(sub_g, v = start_node, mode = "all")[1,]
      } else {
        traversal_result <- custom_dfs(sub_g, start_node)
        visited_nodes <- V(sub_g)$name[traversal_result$order]
        node_distances <- distances(sub_g, v = start_node, mode = "all")[1,]
      }
      
      visited_nodes <- visited_nodes[!is.na(visited_nodes)]
      
      max_viz_nodes <- min(30, length(visited_nodes))
      viz_nodes <- visited_nodes[1:max_viz_nodes]
      viz_g <- induced_subgraph(sub_g, viz_nodes)
      
      incProgress(0.8, detail = "Generating visualizations")
      
      output$seller_plot <- renderPlot({
        ggplot(seller_data, aes(fraud_score, fill = factor(fraud_label))) +
          geom_histogram(bins = 20, alpha = 0.7) +
          labs(
            title = "Seller Fraud Score Distribution",
            x = "Fraud Probability",
            y = "Count",
            fill = "Label"
          ) +
          scale_fill_manual(
            values = c("0" = "lightgreen", "1" = "red"),
            labels = c("0" = "Legitimate", "1" = "Fraudulent")
          ) +
          theme_minimal()
      })
      
      output$seller_metrics <- renderPrint({
        cat("=== SELLER MODEL PERFORMANCE (Test Set) ===\n\n")
        print(seller_cm)
        cat("\n\nFeature Importance:\n")
        print(importance(rf_model))
      })
      
      output$seller_table <- renderDT({
        datatable(seller_data %>% 
                    arrange(desc(fraud_score)) %>%
                    mutate(fraud_score = round(fraud_score, 3)),
                  options = list(pageLength = 10))
      })
      
      output$review_plot <- renderPlot({
        ggplot(review_data, aes(fraud_score, fill = factor(fraud_label))) +
          geom_histogram(bins = 20, alpha = 0.7) +
          labs(
            title = "Review Fraud Score Distribution",
            x = "Fraud Probability",
            y = "Count",
            fill = "Label"
          ) +
          scale_fill_manual(
            values = c("0" = "lightgreen", "1" = "red"),
            labels = c("0" = "Legitimate", "1" = "Fraudulent")
          ) +
          theme_minimal()
      })
      
      output$review_metrics <- renderPrint({
        cat("=== REVIEW MODEL PERFORMANCE (Test Set) ===\n\n")
        print(review_cm)
      })
      
      output$review_table <- renderDT({
        datatable(review_data %>% 
                    arrange(desc(fraud_score)) %>%
                    mutate(fraud_score = round(fraud_score, 3)),
                  options = list(pageLength = 10))
      })
      
      output$graph_plot <- renderPlot({
        depths <- node_distances[V(viz_g)$name]
        max_depth <- max(depths[is.finite(depths)])
        
        layout_matrix <- layout_as_tree(viz_g, root = start_node)
        
        fraud_scores_viz <- sapply(V(viz_g)$name, function(v) {
          score <- V(viz_g)[v]$fraud_score
          if (is.null(score) || length(score) == 0) return(0)
          return(score)
        })
        fraud_colors <- colorRampPalette(c("lightgreen", "yellow", "orange", "red"))(100)
        colors <- fraud_colors[pmin(100, pmax(1, ceiling(fraud_scores_viz * 100)))]
        
        account_ages <- sapply(V(viz_g)$name, function(v) {
          age <- V(viz_g)[v]$account_age
          if (is.null(age) || length(age) == 0) return(50)
          return(age)
        })
        sizes <- 15 + (account_ages / 5)
        
        edge_widths <- E(viz_g)$shared_reviews / 2
        
        plot(
          viz_g,
          layout = layout_matrix,
          vertex.color = colors,
          vertex.size = sizes,
          edge.width = edge_widths,
          vertex.label.cex = 0.7,
          vertex.label.color = "black",
          vertex.label.dist = 1.5,
          edge.arrow.size = 0,
          main = paste(input$graph_algo, "Traversal | Fraud Network | Start:", start_node),
          sub = "Color: Fraud Score | Size: Account Age | Edge Width: Shared Reviews"
        )
        
        legend("topright", 
               legend = c("High Fraud", "Medium Fraud", "Low Fraud"),
               col = c("red", "orange", "lightgreen"),
               pch = 19,
               pt.cex = 1.5,
               bty = "n")
      })
      
      output$graph_text <- renderPrint({
        cat("=== SELLER FRAUD NETWORK ANALYSIS ===\n\n")
        cat("Network Type: Sellers connected by shared suspicious reviews\n")
        cat("Total Sellers in Network:", vcount(g), "\n")
        cat("Total Connections:", ecount(g), "\n")
        cat("Connected Components:", comps$no, "\n")
        cat("Largest Component Size:", length(comp_nodes), "\n\n")
        cat("Start Node:", start_node, 
            sprintf("(Fraud Score: %.2f)", 
                    ifelse(is.null(V(sub_g)[start_node]$fraud_score), 0, 
                           V(sub_g)[start_node]$fraud_score)), "\n")
        cat("Traversal Algorithm:", input$graph_algo, "\n")
        cat("Nodes Visited:", length(visited_nodes), "\n")
        cat("Showing:", max_viz_nodes, "nodes\n")
        cat("Max Depth from Start:", max(node_distances[is.finite(node_distances)]), "\n\n")
        cat("Traversal Order (first 20 sellers):\n")
        for (i in 1:min(20, length(visited_nodes))) {
          seller <- visited_nodes[i]
          fs <- seller_data$fraud_score[seller_data$seller_id == seller]
          cat(sprintf("  %d. %s (Fraud: %.2f)\n", i, seller, fs))
        }
      })
      
      incProgress(1, detail = "Complete!")
      
    })
    
  })
}

shinyApp(ui, server)