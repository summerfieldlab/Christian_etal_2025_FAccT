# Generate Supplemental Figure A.3: Beeswarm plot of token scores by sentiment (AFINN)
# Negative sentiment (red) vs positive sentiment (blue) across all models and prompts.

library(tidyverse)
library(ggbeeswarm)

# --- Configuration ---
script_dir <- tryCatch(
  dirname(rstudioapi::getActiveDocumentContext()$path),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("--file=", args, value = TRUE)
    if (length(file_arg) > 0) {
      dirname(normalizePath(sub("--file=", "", file_arg)))
    } else {
      "."
    }
  }
)
data_dir <- file.path(script_dir, "..", "data")
output_dir <- file.path(script_dir, "output")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Model name mapping (from config)
config <- yaml::yaml.load_file(file.path(script_dir, "..", "config", "reward_models.yaml"))
model_nicknames <- setNames(
  sapply(config, function(m) m$nickname),
  sapply(config, function(m) gsub("/", "--", m$name))
)
model_levels <- unname(sapply(config, function(m) m$nickname))

model_colors <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")

# --- Load and reshape data ---
scores_dir <- file.path(data_dir, "reward_model_scores")
model_keys <- sapply(config, function(m) gsub("/", "--", m$name))

scores_long <- bind_rows(lapply(seq_along(config), function(i) {
  csv_path <- file.path(scores_dir, paste0(model_keys[i], ".csv"))
  df <- read.csv(csv_path, check.names = FALSE)
  df %>%
    pivot_longer(cols = c("greatest", "best", "worst"),
                 names_to = "prompt", values_to = "score") %>%
    filter(!is.na(score)) %>%
    mutate(
      model_name = model_keys[i],
      model_id = model_nicknames[model_keys[i]],
      prompt = str_to_title(prompt),
      token = tolower(token_decoded)
    )
}))
scores_long <- scores_long %>%
  mutate(model_id = factor(model_id, levels = model_levels))

# --- Load AFINN-111 sentiment lexicon ---
afinn <- read.delim(file.path(data_dir, "corpora", "AFINN-111.txt"),
                    header = FALSE, col.names = c("word", "value"),
                    stringsAsFactors = FALSE)

affect_afinn <- scores_long %>%
  inner_join(afinn, by = c("token" = "word"), multiple = "all") %>%
  filter(!is.na(score), !is.na(value)) %>%
  mutate(sentiment = ifelse(value > 0, "Positive", "Negative"),
         sentiment = factor(sentiment, levels = c("Negative", "Positive")))

cat(sprintf("Matched %d token-score-sentiment rows\n", nrow(affect_afinn)))

# --- Beeswarm plot: score distributions by sentiment, faceted by model x prompt ---
# Map model colors to strip label backgrounds (same palette as Figure A2)
names(model_colors) <- model_levels
strip_fills <- model_colors[levels(affect_afinn$model_id)]

p <- ggplot(affect_afinn, aes(x = sentiment, y = score, color = sentiment)) +
  geom_quasirandom(alpha = 0.3, size = 0.5) +
  scale_color_manual(values = c("Negative" = "#B95C5A", "Positive" = "#3A6E97")) +
  facet_grid(model_id ~ prompt, scales = "free_y") +
  theme_classic(base_family = "sans") +
  labs(x = "Sentiment", y = "Reward model score") +
  theme(
    legend.position = "none",
    strip.text.x = element_text(size = 9),
    strip.text.y.right = element_text(size = 9, color = "white", face = "bold"),
    strip.background.y = element_rect(fill = NA, color = NA),
    axis.text.x = element_text(size = 8),
    axis.text.y = element_text(size = 7),
    panel.spacing = unit(0.3, "lines")
  )

# Color each row strip with its model color using ggh4x
library(ggh4x)
p <- p + facet_grid2(model_id ~ prompt, scales = "free_y",
                     strip = strip_themed(
                       background_y = elem_list_rect(fill = strip_fills),
                       text_y = elem_list_text(colour = rep("white", length(strip_fills)))
                     ))

output_path <- file.path(output_dir, "figure_a3_sentiment_beeswarm.png")
ggsave(output_path, plot = p, width = 8, height = 16, dpi = 300)
cat(sprintf("Saved %s\n", output_path))
