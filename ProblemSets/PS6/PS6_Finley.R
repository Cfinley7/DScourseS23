#Spurs and Lakers Game 

library(rvest)

# Scrape the webpage and select the "Team Stats" table
url <- "https://www.espn.com/nba/matchup/_/gameId/401468884"
page <- read_html(url)
table <- html_nodes(page, ".TeamStatsTable")

# Extract the column names from the table header
header <- html_nodes(table, "thead > tr > th") %>% html_text()

# Extract the data from the table rows
data <- html_nodes(table, "tbody tr") %>% html_nodes("td") %>% html_text() %>% matrix(ncol=17, byrow=TRUE) %>% as.data.frame()

# Add headers for the "Spurs" and "Lakers" teams
names(data)[1] <- "Team"
data$Team[data$Team == "SA"] <- "Spurs"
data$Team[data$Team == "LAL"] <- "Lakers"

# Combine the column names and data into a data frame
team_stats <- data.frame(data)

# Set the correct column names
names(team_stats) <- gsub("^team-stats-", "", header)

# Remove unnecessary rows and columns
team_stats <- team_stats[-c(1, 3)]

# Add missing column headers and rename existing headers
names(team_stats)[1] <- "Team"
names(team_stats)[2] <- "Abbreviation"
names(team_stats) <- c("Category", "Total", "FG", "FG%", "3P", "3P%", "FT", "FT%", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV")

# Convert numeric columns to numeric format
team_stats[, 2:ncol(team_stats)] <- sapply(team_stats[, 2:ncol(team_stats)], function(x) ifelse(is.na(as.numeric(x)), x, as.numeric(x)))

# Save the data frame as a CSV file
write.csv(team_stats, "team_stats.csv", row.names = FALSE)

team_stats

#Visualization

#GGplot
library(ggplot2)
library(dplyr)
library(tidyverse)
team_stats <- read.csv("team_stats.csv")
ggplot(team_stats, aes(x = "Lakers", y = "Spurs"))+ geom_point()

#Bar plot
team_data <- team_stats %>% filter(Team %in% c("Spurs", "Lakers"))
# If you wanted total rebounds for both teams' centers
ggplot(team_data, aes(x = Team, y = REB, fill = Team)) +
  geom_bar(stat = "identity", position = "C") +
  labs(x = "", y = "Total Rebounds", title = "Spurs vs Lakers: Total Rebounds") +
  #Coloring for each team
  scale_fill_manual(values = c("#E03A3E", "#552583")) 

#Heatmap 
library(reshape2)
cor_matrix <- cor(team_stats[, 2:ncol(team_stats)])
ggplot(data = melt(cor_matrix), aes(x = "3P%", y = "3P", fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  labs(title = "Correlation Heatmap of Team Statistics",
       x = "Statistic",
       y = "Statistic")
