# Generate Figure 3: Sentiment analysis (panels A, B, C, D)
# Panel A: AFINN sentiment vs reward model score (S-Lla-8B-v0.2, Greatest)
# Panel B: AFINN sentiment vs reward model score (S-Lla-8B-v0.2, Worst)
# Panel C: Slope estimates for sentiment by model, prompt, and valence
# Panel D: Slope estimates for word frequency by model and prompt

library(tidyverse)
library(broom)
library(ggplot2)
library(ggpubr)

# --- Configuration ---
script_dir <- tryCatch(
  dirname(rstudioapi::getActiveDocumentContext()$path),
  error = function(e) {
    # When run via Rscript, use the script's own location
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

model_colors <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")

# Model name mapping: CSV column prefix -> short nickname (from config)
config <- yaml::yaml.load_file(file.path(script_dir, "..", "config", "reward_models.yaml"))
model_nicknames <- setNames(
  sapply(config, function(m) m$nickname),
  sapply(config, function(m) gsub("/", "--", m$name))
)
model_levels <- unname(sapply(config, function(m) m$nickname))

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

# Set model_id as ordered factor
scores_long <- scores_long %>%
  mutate(model_id = factor(model_id, levels = model_levels))

# --- Load AFINN-111 sentiment lexicon ---
afinn <- read.delim(file.path(script_dir, "..", "data", "corpora", "AFINN-111.txt"),
                    header = FALSE, col.names = c("word", "value"),
                    stringsAsFactors = FALSE)

affect_afinn <- scores_long %>%
  inner_join(afinn, by = c("token" = "word"), multiple = "all") %>%
  filter(!is.na(score), !is.na(value))

cat(sprintf("Matched %d token-score-sentiment rows\n", nrow(affect_afinn)))

# --- Panel A: Scatter, S-Lla-8B-v0.2, Greatest ---
p1 <- affect_afinn %>%
  filter(model_id == "S-Lla-8B-v0.2" & prompt == "Greatest") %>%
  ggplot(aes(x = value, y = score, color = value > 0)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = TRUE) +
  theme_classic(base_family = "sans") +
  labs(x = "Sentiment value", y = "Reward model score") +
  scale_color_manual(values = c("#B95C5A", "#3A6E97")) +
  theme(legend.position = "none") +
  ggtitle("Prompt: Greatest") +
  theme(plot.title = element_text(hjust = 0.5, size = 12))

# --- Panel B: Scatter, S-Lla-8B-v0.2, Worst ---
p2 <- affect_afinn %>%
  filter(model_id == "S-Lla-8B-v0.2" & prompt == "Worst") %>%
  ggplot(aes(x = value, y = score, color = value > 0)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = TRUE) +
  theme_classic(base_family = "sans") +
  labs(x = "Sentiment value", y = "Reward model score") +
  scale_color_manual(values = c("#B95C5A", "#3A6E97")) +
  theme(legend.position = "none") +
  ggtitle("Prompt: Worst") +
  theme(plot.title = element_text(hjust = 0.5, size = 12))

# --- Panel C: Slope estimates by model, prompt, valence ---
affect_regression_results <- affect_afinn %>%
  mutate(affect = if_else(value > 0, "Positive", "Negative")) %>%
  filter(!is.na(value)) %>%
  group_by(model_id, prompt, affect) %>%
  do(tidy(lm(score ~ value, data = ., na.rm = TRUE)))

p3 <- affect_regression_results %>%
  filter(term == "value") %>%
  ggplot(aes(x = affect, y = estimate, color = model_id)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  geom_line(aes(group = paste0(model_id, prompt)), color = "grey") +
  geom_point(size = 3) +
  labs(x = "Sentiment", y = "Slope estimate (sentiment)") +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 2, color = "black") +
  stat_summary(fun.data = mean_se, geom = "errorbar", size = 0.5, width = 0.1, color = "black") +
  theme_classic(base_family = "sans") +
  scale_color_manual(values = model_colors) +
  theme(legend.position = "none") +
  facet_wrap(~prompt)

# --- Panel D: Word frequency slope estimates ---
freq <- read.delim(file.path(script_dir, "..", "data", "corpora", "1_1_all_alpha.txt"))

freq_df <- affect_afinn %>%
  left_join(freq, by = c("token" = "Word"), multiple = "all", relationship = "many-to-many") %>%
  mutate(freq = as.numeric(Freq)) %>%
  filter(!is.na(freq) & freq > 0)

freq_regression_results <- freq_df %>%
  group_by(model_id, prompt) %>%
  do(tidy(lm(score ~ log(freq) + value, data = ., na.action = na.exclude)))

p4 <- freq_regression_results %>%
  filter(term == "log(freq)") %>%
  ggplot(aes(x = prompt, y = estimate, color = model_id, group = prompt)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  geom_point(size = 3) +
  stat_summary(fun.data = mean_se, geom = "errorbar", size = 0.4, width = 0.2, color = "black") +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3, color = "black") +
  scale_color_manual(values = model_colors) +
  labs(x = "Prompt", y = "Slope estimate (log(frequency))") +
  theme_classic(base_family = "sans") +
  theme(legend.position = "none")

# --- Save individual PNGs ---
ggsave(file.path(output_dir, "figure_3a_sentiment_greatest.png"),
       plot = p1, width = 4, height = 4, dpi = 300)
cat(sprintf("Saved figure_3a\n"))

ggsave(file.path(output_dir, "figure_3b_sentiment_worst.png"),
       plot = p2, width = 4, height = 4, dpi = 300)
cat(sprintf("Saved figure_3b\n"))

ggsave(file.path(output_dir, "figure_3c_slope_estimates.png"),
       plot = p3, width = 8, height = 4, dpi = 300)
cat(sprintf("Saved figure_3c\n"))

ggsave(file.path(output_dir, "figure_3d_frequency_slopes.png"),
       plot = p4, width = 4, height = 4, dpi = 300)
cat(sprintf("Saved figure_3d\n"))
