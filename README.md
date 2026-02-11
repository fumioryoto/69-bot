# Telegram Bot 69

A Python Telegram bot named **69** with semantic self-learning memory.

## Features
- `/start` welcome
- `/help` commands help
- `/ping` quick bot liveness check
- `/channel_status` show channel indexing status
- `/learn <key> = <value>` store knowledge
- `/forget <key>` remove knowledge
- `/stats` memory stats
- `/retrain` rebuild all embeddings (useful after changing `EMBED_MODEL`)
- `/readcode <path>` read code with line numbers
- `/findfile <query>` find files/posts from indexed channel uploads
- `/healthcheck` run environment and permissions checks
- `/indexreply` index a replied historical message/file (admin)
- `/postchannel <text>` post text to your target channel (admin only)
- `/postfile` repost a replied file/message to channel (admin)
- `/schedulepost YYYY-MM-DD HH:MM | text` schedule channel post (UTC, admin)
- `/listschedule` list scheduled posts
- `/cancelschedule <id>` cancel schedule (admin)
- `/subscribe_news` start automatic cybersecurity/CVE/vulnerability updates
- `/unsubscribe_news` stop automatic updates
- `/newsnow` fetch latest cybersecurity digest now
- `/newsstatus` show news subscription status
- `/newsource` or `/newsources` list where news is fetched from
- `/autonomous_on` enable autonomous monitoring and brief push
- `/autonomous_off` disable autonomous background activity
- `/autonomous_status` show autonomous mode state
- `/admin_add <user_id>`, `/admin_remove <user_id>`, `/admin_list`
- `/topqueries`, `/topfiles`, `/missing_requests`, `/weekly_report`
- `/backupdata`, `/restoredata <path>`
- Free text chat with semantic memory retrieval
- Adaptive personality that mirrors user tone and verbosity over time

## Installation
### 1. Prerequisites
- Python 3.11+ recommended (tested on 3.14 with compatibility patch)
- `pip`
- Internet connection (for RSS/news and optional embedding model download)

### 2. Clone or open project
```powershell
cd "d:\Telegram bot"
```

### 3. Create virtual environment
Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```
Note: `APScheduler` is included in `requirements.txt`, so PTB JobQueue features
(scheduled jobs, autonomous updates, retry queue) will work after install.

### 5. Configure environment
```powershell
copy .env.example .env
```
Open `.env` and set required values:
- `BOT_TOKEN` (required)
- `TARGET_CHANNEL_ID` (required for channel posting commands)
- `ADMIN_USER_IDS` (recommended)

Optional settings:
- `EMBED_MODEL` (default: `all-MiniLM-L6-v2`)
- `NEWS_INTERVAL_MINUTES` (default: `30`)
- `AUTONOMOUS_BRIEF_HOURS` (default: `6`)
- `ALERT_CHAT_ID` (error alerts)
- `RATE_LIMIT_WINDOW_SEC`, `RATE_LIMIT_MAX_ACTIONS`
- `BACKUP_DIR`
- `CHANNEL_INDEX_ACK` (`true` to make bot send an indexing confirmation in channel)

## Telegram Bot Setup (BotFather)
1. Open Telegram and chat with `@BotFather`.
2. Run `/newbot` and create your bot.
3. Copy the bot token and set `BOT_TOKEN` in `.env`.
4. Optional but recommended in BotFather:
- `/setcommands` and paste your command list
- `/setprivacy` -> **Disable** if you want group message visibility
- `/setjoingroups` -> **Enable** if bot must work in groups/channels

## Channel Setup
1. Add the bot to your channel.
2. Promote bot to **Admin** with permissions to:
- Post messages
- Read channel messages/posts (for indexing new channel content)
3. Set `TARGET_CHANNEL_ID`:
- Public channel: `@channel_username`
- Private channel: numeric id like `-1001234567890`
4. Get your Telegram numeric user ID and add to `ADMIN_USER_IDS`.

## Run
```bash
python bot.py
```

## First-time verification
1. In Telegram chat with the bot, run `/start`.
2. Run `/healthcheck` and make sure checks pass.
3. In your channel, post a sample file/message.
4. Run `/findfile <keyword>` from chat.
5. Test posting with `/postchannel hello` (admin only).
6. In channel, run `/ping` and `/channel_status`.

## If Bot Is \"Not Responding\" In Channel
- By default, channel indexing is silent. This is normal.
- Use `/channel_status` in that channel to confirm indexing count.
- Use `/findfile` in private/group chat to search indexed channel files.
- If you want visible confirmation for each indexed channel post, set:
  - `CHANNEL_INDEX_ACK=true` in `.env`
  - then restart the bot.

## Channel Content Assistant
- Add the bot to your channel as admin.
- The bot indexes new channel posts/files automatically from channel updates.
- Users can run `/findfile <query>` to locate content by filename, mime, caption, or semantic similarity.
- `/findfile` supports filters: `type:`, `from:`, `date:YYYY-MM-DD`, `size:>1000000`, `safe:true`, `page:2`.
- `/findfile` also supports inline Prev/Next pagination buttons.
- If nothing matches, the bot performs internet clue search and suggests links where the file/topic may be found.
- Missing searches are saved and auto-notified later when matching channel content appears.
- Use `/postchannel <text>` to post directly into your configured target channel.
- Use `/postfile` by replying to any message/file for quick repost automation.
- Use `/indexreply` to manually backfill important old posts into the index.

## Notes on self-learning
This bot performs embedding-based incremental learning from chat interactions.

- Facts and conversation samples are stored in `knowledge.json`.
- Per-user conversation style profiles are stored in `knowledge.json` and used to adapt reply style.
- Semantic matches use vector similarity instead of only keyword overlap.
- On first run, the embedding model may download files from Hugging Face.
- If model loading fails, the bot falls back to a hash-vector similarity mode.
- Code commands are workspace-scoped (`WORKSPACE_ROOT`) to avoid path traversal outside project files.
- Internet news updates are pulled from curated security RSS feeds and deduplicated by link.
- Chats are auto-subscribed to cyber news on `/start` (you can opt out with `/unsubscribe_news`).
- Autonomous mode is on by default and can be controlled with `/autonomous_on` and `/autonomous_off`.
- Web clue results include trust score + live link status.
- Moderation blocks unsafe extensions for reposting and applies per-user rate limits.

## Quality
- Basic pytest suite is included in `tests/`.
- GitHub Actions CI runs compile + tests on push/PR.
