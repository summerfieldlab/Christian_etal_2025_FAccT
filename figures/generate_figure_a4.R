# Generate Supplemental Figure A.4: AFINN sentiment vs reward model score
# Scatter plots with regression lines, faceted by model (rows) x prompt (columns).

library(tidyverse)
library(ggh4x)

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
names(model_colors) <- model_levels

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
  filter(!is.na(score), !is.na(value))

cat(sprintf("Matched %d token-score-sentiment rows\n", nrow(affect_afinn)))

# --- Plot: sentiment value vs score, colored by positive/negative ---
strip_fills <- model_colors[levels(affect_afinn$model_id)]

p <- ggplot(affect_afinn, aes(x = value, y = score, color = value > 0)) +
  geom_point(alpha = 0.4, size = 0.8) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 0.8) +
  scale_color_manual(values = c("TRUE" = "#3A6E97", "FALSE" = "#B95C5A")) +
  facet_grid2(model_id ~ prompt, scales = "free_y",
              strip = strip_themed(
                background_y = elem_list_rect(fill = strip_fills),
                text_y = elem_list_text(colour = rep("white", length(strip_fills)))
              )) +
  theme_classic(base_family = "sans") +
  labs(x = "AFINN sentiment value", y = "Reward model score") +
  theme(
    legend.position = "none",
    strip.text.x = element_text(size = 9),
    strip.text.y.right = element_text(size = 9, face = "bold"),
    axis.text.x = element_text(size = 8),
    axis.text.y = element_text(size = 7),
    panel.spacing = unit(0.3, "lines")
  )

output_path <- file.path(output_dir, "figure_a4_sentiment_afinn.png")
ggsave(output_path, plot = p, width = 8, height = 16, dpi = 300)
cat(sprintf("Saved %s\n", output_path))
