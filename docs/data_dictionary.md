# Data dictionary

Export of 2026-09-22: 83 tables, 2,427,464 rows, 703 columns. Tables marked *truncated* hold the first 100,000 rows by primary key, so their time windows differ. `telescope_entries_tags` is listed in the manifest but its file was absent from the archive.

## Tables used by the pipeline

| Table | Rows | Columns | created_at window | Role |
|---|---|---|---|---|
| `questions` | 100,000 (truncated) | 16 | 2022-06-29 – 2025-05-21 | Item bank: text, category, language, cumulative correct/fail counts, reports, moderation status (Parts 1, 2) |
| `answers` | 100,000 (truncated) | 6 | 2022-06-29 – 2022-09-03 | Answer options with selection counts (Part 1 ablation) |
| `correct_answers` | 100,000 (truncated) | 5 | 2022-06-29 – 2025-07-25 | Answer key (Parts 1, 2) |
| `cats` | 275 | 7 | 2022-06-04 – 2026-09-14 | Subject categories, Kazakh and Russian names (Parts 1, 2) |
| `group_questions` | 82,968 | 7 | 2025-05-14 – 2026-09-17 | Response-level tournament answers (Part 2) |
| `games` | 100,000 (truncated) | 11 | 2022-04-23 – 2022-07-21 | Duels: players, winner, status (Part 3) |
| `matches` | 100,000 (truncated) | 21 | 2022-06-23 – 2022-07-13 | Duel rounds: category and correctness of each of 3 answers per player (Part 3) |

## All tables

| Table | Rows | Columns | created_at window | Excluded (PII) columns |
|---|---|---|---|---|
| `answers` | 100,000 (truncated) | 6 | 2022-06-29 – 2022-09-03 |  |
| `chat_messages` | 100,000 (truncated) | 7 | 2022-12-07 – 2023-01-15 | text |
| `correct_answers` | 100,000 (truncated) | 5 | 2022-06-29 – 2025-07-25 |  |
| `friends` | 100,000 (truncated) | 6 | 2022-07-08 – 2022-08-21 |  |
| `games` | 100,000 (truncated) | 11 | 2022-04-23 – 2022-07-21 |  |
| `guests` | 100,000 (truncated) | 6 | 2024-01-29 – 2024-07-01 |  |
| `matches` | 100,000 (truncated) | 21 | 2022-06-23 – 2022-07-13 |  |
| `personal_access_tokens` | 100,000 (truncated) | 6 | 2022-02-22 – 2022-08-27 | name;token;abilities |
| `questions` | 100,000 (truncated) | 16 | 2022-06-29 – 2025-05-21 |  |
| `round_reactions` | 100,000 (truncated) | 7 | 2022-09-14 – 2023-02-19 |  |
| `telescope_entries` | 100,000 (truncated) | 6 | 2026-07-01 – 2026-07-01 | uuid;content |
| `telescope_entries_tags` | 100,000 (truncated) — file missing | 2 | – |  |
| `top_weeks` | 100,000 (truncated) | 7 | 2022-11-02 – 2022-11-02 |  |
| `tournament_games` | 100,000 (truncated) | 5 | 2022-10-07 – 2023-05-24 |  |
| `tournament_meetings` | 100,000 (truncated) | 6 | 2022-10-07 – 2022-12-02 |  |
| `users` | 100,000 (truncated) | 16 | 2022-05-24 – 2022-08-27 | name;password;phone;avatar;access_token;device_token;apple_token;email |
| `user_games` | 100,000 (truncated) | 9 | 2022-08-17 – 2022-09-11 |  |
| `user_news` | 100,000 (truncated) | 11 | 2022-11-05 – 2022-11-07 |  |
| `user_results` | 100,000 (truncated) | 15 | 2022-05-24 – 2022-08-27 |  |
| `group_questions` | 82,968 | 7 | 2025-05-14 – 2026-09-17 |  |
| `chat_message_files` | 82,218 | 6 | 2022-12-07 – 2026-09-19 |  |
| `tournament_users` | 72,979 | 10 | 2022-10-07 – 2026-09-22 |  |
| `baskets` | 49,025 | 6 | 2022-09-11 – 2026-09-15 |  |
| `basket_products` | 48,885 | 7 | 2022-09-14 – 2026-09-15 | user_name;user_phone;address;index |
| `chats` | 45,685 | 3 | 2022-12-07 – 2026-09-21 |  |
| `chat_participants` | 45,483 | 7 | 2026-03-17 – 2026-09-21 |  |
| `big_tournament_users` | 33,240 | 9 | 2024-01-12 – 2026-09-16 |  |
| `tournaments` | 18,724 | 7 | 2022-10-07 – 2026-09-22 |  |
| `report_gamers` | 8,186 | 8 | 2023-04-22 – 2026-09-01 |  |
| `news_resources` | 8,055 | 11 | 2024-06-13 – 2024-07-25 |  |
| `black_lists` | 7,948 | 5 | 2023-08-17 – 2026-09-22 |  |
| `top_cups` | 6,891 | 7 | 2023-05-01 – 2026-07-10 |  |
| `payments` | 3,190 | 10 | 2022-08-17 – 2026-06-30 |  |
| `group_meetings` | 3,032 | 9 | 2025-05-14 – 2026-09-17 |  |
| `group_users` | 3,032 | 9 | 2025-05-14 – 2026-09-17 |  |
| `stories_views` | 2,813 | 6 | 2024-05-13 – 2026-09-22 |  |
| `games_statistics` | 1,473 | 8 | 2022-09-03 – 2026-09-22 |  |
| `report_users` | 725 | 5 | 2023-04-22 – 2023-07-31 |  |
| `groups` | 377 | 21 | 2023-08-24 – 2026-09-12 |  |
| `you_knows` | 365 | 9 | 2024-04-10 – 2024-04-10 |  |
| `cats` | 275 | 7 | 2022-06-04 – 2026-09-14 |  |
| `lang_cats` | 271 | 5 | 2023-06-17 – 2026-07-01 |  |
| `products` | 248 | 18 | 2022-09-14 – 2024-12-10 |  |
| `exports` | 236 | 5 | 2022-12-25 – 2024-12-06 |  |
| `group_cats` | 174 | 6 | 2024-05-28 – 2026-09-12 |  |
| `big_tournament_offers` | 162 | 8 | 2024-02-10 – 2026-09-16 |  |
| `news` | 94 | 14 | 2022-11-05 – 2026-05-04 |  |
| `migrations` | 86 | 3 | – |  |
| `product_partners` | 70 | 6 | 2023-02-15 – 2023-10-11 |  |
| `ads_views` | 67 | 7 | 2024-05-17 – 2026-08-24 |  |
| `auctions` | 62 | 6 | 2023-04-01 – 2023-09-22 |  |
| `app_feedback` | 52 | 5 | 2022-07-18 – 2022-09-05 | text |
| `notifications` | 47 | 10 | 2024-02-12 – 2025-04-12 |  |
| `article_statistics` | 43 | 8 | 2022-11-24 – 2023-06-13 |  |
| `failed_jobs` | 38 | 4 | – | uuid;payload;exception |
| `stages` | 34 | 8 | 2024-03-01 – 2026-09-12 |  |
| `bots` | 30 | 7 | 2023-08-19 – 2026-03-17 |  |
| `stories` | 28 | 14 | 2024-05-13 – 2024-10-25 | number |
| `big_tournaments` | 26 | 26 | 2023-08-24 – 2026-09-12 |  |
| `matchmakings` | 23 | 5 | – |  |
| `cities` | 21 | 5 | 2025-05-31 – 2025-05-31 |  |
| `pushes` | 20 | 11 | 2024-05-04 – 2025-06-08 |  |
| `motivations` | 15 | 5 | 2023-04-24 – 2023-05-03 |  |
| `invites` | 12 | 7 | 2026-09-15 – 2026-09-22 | code |
| `partners` | 8 | 9 | 2023-02-15 – 2023-07-19 |  |
| `tarifs` | 7 | 10 | 2024-04-05 – 2024-04-05 |  |
| `product_cats` | 5 | 8 | – |  |
| `ads` | 4 | 12 | 2024-05-17 – 2024-09-02 |  |
| `banners` | 4 | 11 | 2024-07-02 – 2024-09-02 |  |
| `phones` | 2 | 4 | – | number |
| `fairs` | 1 | 7 | 2023-02-07 – 2023-02-07 |  |
| `feedback` | 1 | 3 | – | phone;email;telegram;instagram;vk |
| `for_partners` | 1 | 8 | 2023-02-04 – 2023-02-04 |  |
| `global_statistics` | 1 | 10 | 2022-09-07 – 2022-09-07 |  |
| `info_blocs` | 1 | 8 | 2023-05-21 – 2023-05-21 |  |
| `version_apps` | 1 | 9 | 2022-03-18 – 2022-03-18 | phone |
| `active_games` | 0 (empty) | 5 | – |  |
| `irl_decision_events` | 0 (empty) | 29 | – | session_uuid |
| `jobs` | 0 (empty) | 6 | – | payload |
| `notification_users` | 0 (empty) | 11 | – |  |
| `product_media` | 0 (empty) | 6 | – |  |
| `question_jobs` | 0 (empty) | 8 | – |  |
| `telescope_monitoring` | 0 (empty) | 1 | – |  |
