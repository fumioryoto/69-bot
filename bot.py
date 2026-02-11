import json
import logging
import os
import re
import math
import hashlib
import asyncio
import calendar
import shutil
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List

import time

from dotenv import load_dotenv
import feedparser
from duckduckgo_search import DDGS
import requests
from sentence_transformers import SentenceTransformer
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("bot69")

BOT_NAME = "69"
DATA_FILE = Path("knowledge.json")
MAX_MEMORY_ITEMS = 2000
MIN_FACT_SIM = 0.45
MIN_SAMPLE_SIM = 0.40
PERSONA_NAME = "69"
PERSONA_SIGNATURE = " - 69"
MAX_CODE_CHARS = 12000
MAX_TELEGRAM_REPLY_CHARS = 3500
MAX_NEWS_ITEMS_PER_PUSH = 6
NEWS_INTERVAL_MINUTES = int(os.getenv("NEWS_INTERVAL_MINUTES", "30"))
AUTONOMOUS_BRIEF_HOURS = int(os.getenv("AUTONOMOUS_BRIEF_HOURS", "6"))
MAX_NEWS_SEEN_LINKS = 2000
MAX_CHANNEL_INDEX_ITEMS = 5000
TARGET_CHANNEL_ID = os.getenv("TARGET_CHANNEL_ID", "").strip()
ADMIN_USER_IDS = {
    int(x.strip())
    for x in os.getenv("ADMIN_USER_IDS", "").split(",")
    if x.strip().isdigit()
}
ALERT_CHAT_ID = os.getenv("ALERT_CHAT_ID", "").strip()
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "30"))
RATE_LIMIT_MAX_ACTIONS = int(os.getenv("RATE_LIMIT_MAX_ACTIONS", "12"))
MAX_OPEN_MISSING_REQUESTS = 500
BACKUP_DIR = Path(os.getenv("BACKUP_DIR", "backups"))
CHANNEL_INDEX_ACK = os.getenv("CHANNEL_INDEX_ACK", "false").strip().lower() in {"1", "true", "yes"}
UNSAFE_EXTENSIONS = {".exe", ".bat", ".cmd", ".ps1", ".scr", ".com", ".msi", ".vbs"}
TRUSTED_SOURCE_SCORES = {
    "nvd.nist.gov": 1.0,
    "cisa.gov": 1.0,
    "github.com": 0.8,
    "microsoft.com": 0.8,
    "ubuntu.com": 0.8,
    "bleepingcomputer.com": 0.7,
    "thehackernews.com": 0.6,
    "exploit-db.com": 0.6,
}
NEWS_FEEDS = [
    ("CISA Advisories", "https://www.cisa.gov/cybersecurity-advisories/all.xml"),
    ("Exploit-DB", "https://www.exploit-db.com/rss.xml"),
    ("The Hacker News", "https://feeds.feedburner.com/TheHackersNews"),
    ("BleepingComputer", "https://www.bleepingcomputer.com/feed/"),
]
RATE_LIMIT_STATE: Dict[int, List[float]] = {}
FIND_SESSIONS: Dict[str, Dict[str, Any]] = {}


class Embedder:
    def __init__(self) -> None:
        self.backend = "hash-fallback"
        self.model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        self.model = None
        if os.getenv("DISABLE_EMBED_MODEL", "").strip().lower() in {"1", "true", "yes"}:
            logger.info("Embedding model disabled by DISABLE_EMBED_MODEL")
            return
        try:
            self.model = SentenceTransformer(self.model_name)
            self.backend = f"sentence-transformers:{self.model_name}"
        except Exception as exc:
            logger.warning("Embedding model unavailable, using fallback vectors: %s", exc)

    def embed(self, text: str) -> List[float]:
        cleaned = text.strip()
        if not cleaned:
            return []
        if self.model is not None:
            vec = self.model.encode(cleaned, normalize_embeddings=True)
            return [float(x) for x in vec.tolist()]
        return self._hash_embed(cleaned)

    def _hash_embed(self, text: str, dim: int = 256) -> List[float]:
        words = re.findall(r"\w+", text.lower())
        if not words:
            return []
        vec = [0.0] * dim
        for token in words:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:2], "big") % dim
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return []
        return [x / norm for x in vec]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def with_line_numbers(text: str, start_at: int = 1) -> str:
    lines = text.splitlines()
    return "\n".join(f"{i + start_at:4d}: {line}" for i, line in enumerate(lines))


def detect_language(text: str) -> str:
    # Lightweight heuristic to avoid adding heavyweight translation dependencies.
    if re.search(r"[\u0980-\u09FF]", text):
        return "bn"
    return "en"


def get_domain(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url.lower().strip())
    return m.group(1) if m else ""


def trust_score(url: str) -> float:
    domain = get_domain(url)
    for host, score in TRUSTED_SOURCE_SCORES.items():
        if domain.endswith(host):
            return score
    return 0.3


def is_safe_extension(file_name: str) -> bool:
    ext = Path(file_name.lower()).suffix
    return ext not in UNSAFE_EXTENSIONS


def persona_reply(core: str, profile: Dict[str, Any] | None = None) -> str:
    text = core.strip()
    if not text:
        text = "I am listening."

    profile = profile or {}
    tone = str(profile.get("tone", "neutral"))
    verbosity = str(profile.get("verbosity", "medium"))
    emoji_pref = float(profile.get("emoji_pref", 0.0))
    exclaim_pref = float(profile.get("exclaim_pref", 0.0))

    if verbosity == "short":
        first_line = text.splitlines()[0].strip()
        text = first_line[:160] if first_line else text[:160]
    elif verbosity == "medium" and len(text) > 450:
        text = text[:450].rstrip() + "..."

    if tone == "formal":
        text = text.replace("I do not", "I do not").replace("Locked in", "Stored")
    elif tone == "casual":
        if not text.lower().startswith(("yo", "ok", "got it")):
            text = f"Yo, {text}"

    if tone != "formal" and exclaim_pref >= 0.45 and not text.endswith(("!", "?")):
        text += "!"
    if tone != "formal" and emoji_pref >= 0.5:
        text += " :)"
    if not text.endswith(PERSONA_SIGNATURE):
        text += PERSONA_SIGNATURE
    return text


class MemoryStore:
    def __init__(self, path: Path, embedder: Embedder) -> None:
        self.path = path
        self.embedder = embedder
        self.data: Dict[str, Any] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to load %s, starting fresh: %s", self.path, exc)
                self.data = {}
        self._migrate()

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _migrate(self) -> None:
        changed = False

        if not isinstance(self.data, dict):
            self.data = {}
            changed = True
        if not isinstance(self.data.get("facts"), dict):
            self.data["facts"] = {}
            changed = True
        if not isinstance(self.data.get("samples"), list):
            self.data["samples"] = []
            changed = True
        if not isinstance(self.data.get("profiles"), dict):
            self.data["profiles"] = {}
            changed = True
        if not isinstance(self.data.get("news"), dict):
            self.data["news"] = {}
            changed = True
        if not isinstance(self.data["news"].get("subscribers"), list):
            self.data["news"]["subscribers"] = []
            changed = True
        if not isinstance(self.data["news"].get("seen_links"), list):
            self.data["news"]["seen_links"] = []
            changed = True
        if not isinstance(self.data["news"].get("autonomous_enabled"), bool):
            self.data["news"]["autonomous_enabled"] = True
            changed = True
        if not isinstance(self.data.get("channel_index"), dict):
            self.data["channel_index"] = {}
            changed = True
        if not isinstance(self.data.get("auth"), dict):
            self.data["auth"] = {}
            changed = True
        if not isinstance(self.data["auth"].get("admins"), list):
            self.data["auth"]["admins"] = sorted(list(ADMIN_USER_IDS))
            changed = True
        if not isinstance(self.data.get("analytics"), dict):
            self.data["analytics"] = {}
            changed = True
        if not isinstance(self.data["analytics"].get("queries"), list):
            self.data["analytics"]["queries"] = []
            changed = True
        if not isinstance(self.data["analytics"].get("abuse_logs"), list):
            self.data["analytics"]["abuse_logs"] = []
            changed = True
        if not isinstance(self.data.get("missing_requests"), list):
            self.data["missing_requests"] = []
            changed = True
        if not isinstance(self.data.get("scheduled_posts"), list):
            self.data["scheduled_posts"] = []
            changed = True
        if not isinstance(self.data.get("retry_queue"), list):
            self.data["retry_queue"] = []
            changed = True

        migrated_facts = {}
        for k, v in self.data.get("facts", {}).items():
            if isinstance(v, str):
                emb = self.embedder.embed(f"{k}: {v}")
                migrated_facts[k] = {"value": v, "embedding": emb}
                changed = True
                continue
            if isinstance(v, dict):
                value = str(v.get("value", ""))
                embedding = v.get("embedding")
                if not isinstance(embedding, list):
                    embedding = self.embedder.embed(f"{k}: {value}")
                    changed = True
                migrated_facts[k] = {"value": value, "embedding": embedding}
                continue
            migrated_facts[k] = {"value": str(v), "embedding": self.embedder.embed(f"{k}: {v}")}
            changed = True
        self.data["facts"] = migrated_facts

        migrated_samples = []
        for item in self.data.get("samples", []):
            if not isinstance(item, dict):
                changed = True
                continue
            user_text = str(item.get("user", ""))
            bot_text = str(item.get("bot", ""))
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                embedding = self.embedder.embed(user_text)
                changed = True
            migrated_samples.append(
                {"user": user_text, "bot": bot_text, "embedding": embedding}
            )
        self.data["samples"] = migrated_samples

        if changed:
            self._save()
    def _default_profile(self) -> Dict[str, float]:
        return {
            "messages": 0.0,
            "avg_words": 0.0,
            "formal_score": 0.0,
            "casual_score": 0.0,
            "emoji_pref": 0.0,
            "exclaim_pref": 0.0,
            "question_pref": 0.0,
        }

    def get_adaptive_profile(self, user_key: str) -> Dict[str, Any]:
        p = self.data["profiles"].get(user_key, self._default_profile())
        tone = "neutral"
        if p["formal_score"] > p["casual_score"] + 0.15:
            tone = "formal"
        elif p["casual_score"] > p["formal_score"] + 0.15:
            tone = "casual"

        verbosity = "medium"
        if p["avg_words"] <= 5:
            verbosity = "short"
        elif p["avg_words"] >= 18:
            verbosity = "long"

        return {
            "tone": tone,
            "verbosity": verbosity,
            "emoji_pref": p["emoji_pref"],
            "exclaim_pref": p["exclaim_pref"],
        }

    def update_profile_from_text(self, user_key: str, text: str) -> None:
        profile = self.data["profiles"].get(user_key, self._default_profile())
        msg_count = int(profile["messages"]) + 1

        words = re.findall(r"\w+", text.lower())
        word_count = len(words)

        formal_markers = {"please", "kindly", "could", "would", "thank", "thanks"}
        casual_markers = {"yo", "bro", "dude", "lol", "haha", "pls", "thx", "hey"}

        formal_hit = 1.0 if any(token in formal_markers for token in words) else 0.0
        casual_hit = 1.0 if any(token in casual_markers for token in words) else 0.0
        emoji_hit = 1.0 if re.search(r"[\U0001F300-\U0001FAFF]", text) else 0.0
        exclaim_hit = 1.0 if "!" in text else 0.0
        question_hit = 1.0 if "?" in text else 0.0

        profile["messages"] = float(msg_count)
        profile["avg_words"] = (
            (float(profile["avg_words"]) * (msg_count - 1) + word_count) / msg_count
        )
        profile["formal_score"] = (
            (float(profile["formal_score"]) * (msg_count - 1) + formal_hit) / msg_count
        )
        profile["casual_score"] = (
            (float(profile["casual_score"]) * (msg_count - 1) + casual_hit) / msg_count
        )
        profile["emoji_pref"] = (
            (float(profile["emoji_pref"]) * (msg_count - 1) + emoji_hit) / msg_count
        )
        profile["exclaim_pref"] = (
            (float(profile["exclaim_pref"]) * (msg_count - 1) + exclaim_hit) / msg_count
        )
        profile["question_pref"] = (
            (float(profile["question_pref"]) * (msg_count - 1) + question_hit) / msg_count
        )

        self.data["profiles"][user_key] = profile
        self._save()

    def learn_fact(self, key: str, value: str) -> None:
        key_n = key.lower().strip()
        value_n = value.strip()
        self.data["facts"][key_n] = {
            "value": value_n,
            "embedding": self.embedder.embed(f"{key_n}: {value_n}"),
        }
        self._save()

    def forget_fact(self, key: str) -> bool:
        removed = self.data["facts"].pop(key.lower().strip(), None)
        if removed is None:
            return False
        self._save()
        return True

    def get_fact(self, key: str) -> str | None:
        item = self.data["facts"].get(key.lower().strip())
        if not item:
            return None
        return item.get("value")

    def semantic_facts(self, text: str, limit: int = 3) -> List[Dict[str, str | float]]:
        q_emb = self.embedder.embed(text)
        if not q_emb:
            return []
        scored = []
        query_words = set(re.findall(r'\w+', text.lower()))
        for key, item in self.data["facts"].items():
            emb = item.get("embedding", [])
            emb_score = cosine_similarity(q_emb, emb)
            fact_words = set(re.findall(r'\w+', key.lower()))
            keyword_sim = len(query_words & fact_words) / max(len(query_words | fact_words), 1)
            combined_score = 0.7 * emb_score + 0.3 * keyword_sim
            if combined_score >= MIN_FACT_SIM:
                scored.append({"key": key, "value": item.get("value", ""), "score": combined_score})
        scored.sort(key=lambda x: float(x["score"]), reverse=True)
        # Cross-encoder re-ranking
        try:
            from sentence_transformers import CrossEncoder
            cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
            pairs = [[text, f"{item['key']}: {item['value']}"] for item in scored[:10]]
            ce_scores = cross_encoder.predict(pairs)
            for idx, item in enumerate(scored[:10]):
                item['ce_score'] = float(ce_scores[idx])
            scored[:10] = sorted(scored[:10], key=lambda x: x.get('ce_score', 0), reverse=True)
            return scored[:limit]
        except Exception:
            return scored[:limit]

    def learn_sample(self, user_text: str, bot_text: str) -> None:
        self.data["samples"].append(
            {
                "user": user_text.strip()[:500],
                "bot": bot_text.strip()[:500],
                "embedding": self.embedder.embed(user_text),
                "timestamp": time.time(),
            }
        )
        if len(self.data["samples"]) > MAX_MEMORY_ITEMS:
            self.data["samples"] = self.data["samples"][-MAX_MEMORY_ITEMS:]
        self._save()

    def similar_responses(self, text: str, limit: int = 3) -> List[str]:
        q_emb = self.embedder.embed(text)
        if not q_emb:
            return []
        scored = []
        now = time.time()
        for item in self.data["samples"]:
            emb_score = cosine_similarity(q_emb, item.get("embedding", []))
            timestamp = item.get("timestamp", now)
            age_days = (now - timestamp) / 86400
            recency_bonus = 1.0 if age_days < 7 else max(0.5, 1.0 - age_days/365)
            final_score = emb_score * recency_bonus
            if final_score >= MIN_SAMPLE_SIM:
                scored.append((final_score, item.get("bot", "")))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [resp for _, resp in scored[:limit] if resp]

    def stats(self) -> str:
        return (
            f"Facts: {len(self.data['facts'])}\n"
            f"Conversation samples: {len(self.data['samples'])}\n"
            f"Adaptive profiles: {len(self.data['profiles'])}\n"
            f"Embedding backend: {self.embedder.backend}"
        )

    def retrain_embeddings(self) -> Dict[str, int]:
        fact_count = 0
        sample_count = 0

        for key, item in self.data["facts"].items():
            value = str(item.get("value", ""))
            item["embedding"] = self.embedder.embed(f"{key}: {value}")
            fact_count += 1

        for item in self.data["samples"]:
            user_text = str(item.get("user", ""))
            item["embedding"] = self.embedder.embed(user_text)
            sample_count += 1

        self._save()
        return {"facts": fact_count, "samples": sample_count}

    def subscribe_news(self, chat_id: int) -> bool:
        key = str(chat_id)
        subs = self.data["news"]["subscribers"]
        if key in subs:
            return False
        subs.append(key)
        self._save()
        return True

    def unsubscribe_news(self, chat_id: int) -> bool:
        key = str(chat_id)
        subs = self.data["news"]["subscribers"]
        if key not in subs:
            return False
        subs.remove(key)
        self._save()
        return True

    def news_subscribers(self) -> List[int]:
        out: List[int] = []
        for item in self.data["news"]["subscribers"]:
            try:
                out.append(int(item))
            except Exception:
                continue
        return out

    def is_autonomous_enabled(self) -> bool:
        return bool(self.data["news"].get("autonomous_enabled", True))

    def set_autonomous_enabled(self, enabled: bool) -> None:
        self.data["news"]["autonomous_enabled"] = bool(enabled)
        self._save()

    def seen_news_links(self) -> set[str]:
        return set(str(x) for x in self.data["news"]["seen_links"])

    def remember_news_links(self, links: List[str]) -> None:
        if not links:
            return
        seen = self.data["news"]["seen_links"]
        seen.extend(links)
        deduped = list(dict.fromkeys(seen))
        self.data["news"]["seen_links"] = deduped[-MAX_NEWS_SEEN_LINKS:]
        self._save()

    def admin_ids(self) -> List[int]:
        out: List[int] = []
        for x in self.data["auth"].get("admins", []):
            try:
                out.append(int(x))
            except Exception:
                continue
        return sorted(list(set(out)))

    def is_admin(self, user_id: int | None) -> bool:
        if user_id is None:
            return False
        admins = self.admin_ids()
        if not admins and ADMIN_USER_IDS:
            return user_id in ADMIN_USER_IDS
        if not admins and not ADMIN_USER_IDS:
            return True
        return user_id in admins

    def add_admin(self, user_id: int) -> bool:
        admins = set(self.admin_ids())
        if user_id in admins:
            return False
        admins.add(user_id)
        self.data["auth"]["admins"] = sorted(list(admins))
        self._save()
        return True

    def remove_admin(self, user_id: int) -> bool:
        admins = set(self.admin_ids())
        if user_id not in admins:
            return False
        admins.remove(user_id)
        self.data["auth"]["admins"] = sorted(list(admins))
        self._save()
        return True

    def record_query(self, chat_id: int, user_id: int, query: str, found_count: int) -> None:
        rows = self.data["analytics"]["queries"]
        rows.append(
            {
                "chat_id": int(chat_id),
                "user_id": int(user_id),
                "query": query[:200],
                "found_count": int(found_count),
                "ts": datetime.now(tz=timezone.utc).isoformat(),
            }
        )
        self.data["analytics"]["queries"] = rows[-5000:]
        self._save()

    def record_abuse(self, user_id: int, reason: str) -> None:
        logs = self.data["analytics"]["abuse_logs"]
        logs.append(
            {
                "user_id": int(user_id),
                "reason": reason[:200],
                "ts": datetime.now(tz=timezone.utc).isoformat(),
            }
        )
        self.data["analytics"]["abuse_logs"] = logs[-2000:]
        self._save()

    def add_missing_request(self, chat_id: int, user_id: int, query: str) -> None:
        rows = self.data["missing_requests"]
        rows.append(
            {
                "id": int(datetime.now(tz=timezone.utc).timestamp() * 1000),
                "chat_id": int(chat_id),
                "user_id": int(user_id),
                "query": query[:200],
                "fulfilled": False,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "fulfilled_at": "",
                "matched_link": "",
            }
        )
        self.data["missing_requests"] = rows[-MAX_OPEN_MISSING_REQUESTS:]
        self._save()

    def open_missing_requests(self) -> List[Dict[str, Any]]:
        return [x for x in self.data["missing_requests"] if not bool(x.get("fulfilled", False))]

    def fulfill_missing_requests(self, text_blob: str, matched_link: str) -> List[Dict[str, Any]]:
        q = text_blob.lower()
        fulfilled: List[Dict[str, Any]] = []
        for row in self.data["missing_requests"]:
            if row.get("fulfilled", False):
                continue
            query = str(row.get("query", "")).lower().strip()
            if not query:
                continue
            if query in q or any(tok in q for tok in query.split()[:4]):
                row["fulfilled"] = True
                row["fulfilled_at"] = datetime.now(tz=timezone.utc).isoformat()
                row["matched_link"] = matched_link
                fulfilled.append(dict(row))
        if fulfilled:
            self._save()
        return fulfilled

    def top_queries(self, days: int = 7, limit: int = 10) -> List[tuple[str, int]]:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
        counts: Dict[str, int] = {}
        for row in self.data["analytics"]["queries"]:
            try:
                ts = datetime.fromisoformat(str(row.get("ts")))
            except Exception:
                continue
            if ts < cutoff:
                continue
            q = str(row.get("query", "")).strip()
            if not q:
                continue
            counts[q] = counts.get(q, 0) + 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:limit]

    def top_files(self, limit: int = 10) -> List[tuple[str, int]]:
        counts: Dict[str, int] = {}
        for _, rows in self.data["channel_index"].items():
            for row in rows:
                name = str(row.get("file_name", "")).strip() or "(text)"
                counts[name] = counts.get(name, 0) + 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:limit]

    def add_scheduled_post(self, post: Dict[str, Any]) -> None:
        rows = self.data["scheduled_posts"]
        rows.append(post)
        self.data["scheduled_posts"] = rows[-500:]
        self._save()

    def remove_scheduled_post(self, post_id: int) -> bool:
        rows = self.data["scheduled_posts"]
        new_rows = [x for x in rows if int(x.get("id", 0)) != int(post_id)]
        if len(new_rows) == len(rows):
            return False
        self.data["scheduled_posts"] = new_rows
        self._save()
        return True

    def list_scheduled_posts(self) -> List[Dict[str, Any]]:
        return list(self.data["scheduled_posts"])

    def push_retry(self, payload: Dict[str, Any]) -> None:
        rows = self.data["retry_queue"]
        rows.append(payload)
        self.data["retry_queue"] = rows[-500:]
        self._save()

    def pop_retry_batch(self, limit: int = 20) -> List[Dict[str, Any]]:
        rows = self.data["retry_queue"]
        batch = rows[:limit]
        self.data["retry_queue"] = rows[limit:]
        if batch:
            self._save()
        return batch

    def upsert_channel_item(self, channel_key: str, item: Dict[str, Any]) -> None:
        bucket = self.data["channel_index"].get(channel_key, [])
        message_id = int(item.get("message_id", 0))
        file_unique_id = str(item.get("file_unique_id", ""))
        updated = False
        for idx, row in enumerate(bucket):
            if int(row.get("message_id", 0)) == message_id:
                bucket[idx] = item
                updated = True
                break
            if file_unique_id and str(row.get("file_unique_id", "")) == file_unique_id:
                # Keep latest message reference for duplicate file re-post.
                bucket[idx] = item
                updated = True
                break
        if not updated:
            bucket.append(item)
        bucket = sorted(bucket, key=lambda x: int(x.get("message_id", 0)), reverse=True)
        self.data["channel_index"][channel_key] = bucket[:MAX_CHANNEL_INDEX_ITEMS]
        self._save()

    def search_channel_items(
        self,
        query: str,
        channel_key: str | None = None,
        limit: int = 6,
        filters: Dict[str, str] | None = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        filters = filters or {}
        q_emb = self.embedder.embed(query)
        channels = [channel_key] if channel_key else list(self.data["channel_index"].keys())
        scored: List[Dict[str, Any]] = []
        for ch in channels:
            if ch is None:
                continue
            for row in self.data["channel_index"].get(ch, []):
                if filters.get("from") and filters["from"].lower() not in str(ch).lower():
                    continue
                if filters.get("type") and filters["type"].lower() not in str(row.get("kind", "")).lower():
                    continue
                if filters.get("safe") == "true" and not bool(row.get("is_safe_ext", True)):
                    continue
                if filters.get("date"):
                    date_text = str(row.get("date", ""))[:10]
                    if date_text != filters["date"]:
                        continue
                if filters.get("size"):
                    size = int(row.get("file_size", 0))
                    expr = filters["size"]
                    if expr.startswith(">") and size <= int(expr[1:]):
                        continue
                    if expr.startswith("<") and size >= int(expr[1:]):
                        continue
                emb = row.get("embedding", [])
                score = cosine_similarity(q_emb, emb) if q_emb else 0.0
                text = str(row.get("search_text", "")).lower()
                if query.lower() in text:
                    score = max(score, 0.85)
                if score >= 0.25:
                    out = dict(row)
                    out["score"] = score
                    scored.append(out)
        scored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return scored[offset:offset + limit]

    def channel_item_count(self, channel_key: str) -> int:
        return len(self.data["channel_index"].get(channel_key, []))


@dataclass
class CodeToolResult:
    ok: bool
    text: str


class CodeAssistant:
    def __init__(self) -> None:
        self.root = Path(os.getenv("WORKSPACE_ROOT", ".")).resolve()

    def _resolve(self, rel_path: str) -> Path:
        path = (self.root / rel_path).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError("Path escapes workspace root.")
        return path

    def read_code(self, rel_path: str, max_chars: int = MAX_CODE_CHARS) -> CodeToolResult:
        try:
            path = self._resolve(rel_path)
            if not path.exists() or not path.is_file():
                return CodeToolResult(False, "File not found.")
            content = path.read_text(encoding="utf-8", errors="replace")
            if len(content) > max_chars:
                content = content[:max_chars] + "\n... [truncated]"
            numbered = with_line_numbers(content)
            return CodeToolResult(True, f"File: {rel_path}\n{numbered}")
        except Exception as exc:
            return CodeToolResult(False, f"Read failed: {exc}")


@dataclass
class NewsItem:
    source: str
    title: str
    link: str
    published: datetime
    category: str


class NewsWatcher:
    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory

    def _parse_date(self, entry: Any) -> datetime:
        for key in ("published_parsed", "updated_parsed", "created_parsed"):
            stamp = entry.get(key)
            if stamp:
                return datetime.fromtimestamp(calendar.timegm(stamp), tz=timezone.utc)
        return datetime.now(tz=timezone.utc)

    def _category(self, text: str) -> str:
        lower = text.lower()
        if "cve-" in lower:
            return "CVE"
        if any(word in lower for word in ["rce", "zero-day", "exploit", "vulnerability"]):
            return "Vulnerability"
        if any(word in lower for word in ["apt", "ransomware", "malware", "phishing"]):
            return "Threat Trend"
        return "Cyber News"

    def _fetch_sync(self, per_feed_limit: int = 15) -> List[NewsItem]:
        items: List[NewsItem] = []
        for source, url in NEWS_FEEDS:
            try:
                parsed = feedparser.parse(url)
                for entry in parsed.entries[:per_feed_limit]:
                    title = str(entry.get("title", "")).strip()
                    link = str(entry.get("link", "")).strip()
                    summary = str(entry.get("summary", ""))
                    if not title or not link:
                        continue
                    items.append(
                        NewsItem(
                            source=source,
                            title=title,
                            link=link,
                            published=self._parse_date(entry),
                            category=self._category(f"{title} {summary}"),
                        )
                    )
            except Exception as exc:
                logger.warning("News feed failed (%s): %s", source, exc)

        dedup_by_link: Dict[str, NewsItem] = {}
        for item in items:
            dedup_by_link[item.link] = item
        unique = list(dedup_by_link.values())
        unique.sort(key=lambda x: x.published, reverse=True)
        return unique

    async def fetch_latest(self, per_feed_limit: int = 15) -> List[NewsItem]:
        return await asyncio.to_thread(self._fetch_sync, per_feed_limit)

    async def fetch_fresh(self, max_items: int = MAX_NEWS_ITEMS_PER_PUSH) -> List[NewsItem]:
        latest = await self.fetch_latest()
        seen = self.memory.seen_news_links()
        fresh: List[NewsItem] = []
        for item in latest:
            if item.link in seen:
                continue
            fresh.append(item)
            if len(fresh) >= max_items:
                break
        self.memory.remember_news_links([item.link for item in fresh])
        return fresh

    def format_digest(self, items: List[NewsItem], title: str) -> str:
        if not items:
            return f"{title}\nNo new items found."
        lines = [title]
        for item in items:
            when = item.published.strftime("%Y-%m-%d %H:%M UTC")
            lines.append(
                f"- [{item.category}] {item.title}\n"
                f"  Source: {item.source}\n"
                f"  Time: {when}\n"
                f"  {item.link}"
            )
        return "\n".join(lines)

    def format_trend_brief(self, items: List[NewsItem], title: str = "Autonomous Trend Brief") -> str:
        if not items:
            return f"{title}\nNo current trend data."
        by_category: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        for item in items:
            by_category[item.category] = by_category.get(item.category, 0) + 1
            by_source[item.source] = by_source.get(item.source, 0) + 1
        cat_rank = sorted(by_category.items(), key=lambda x: x[1], reverse=True)
        src_rank = sorted(by_source.items(), key=lambda x: x[1], reverse=True)
        lines = [title, "Top categories:"]
        for name, count in cat_rank[:3]:
            lines.append(f"- {name}: {count}")
        lines.append("Top sources:")
        for name, count in src_rank[:3]:
            lines.append(f"- {name}: {count}")
        lines.append("Latest highlights:")
        for item in items[:3]:
            lines.append(f"- [{item.category}] {item.title}\n  {item.link}")
        return "\n".join(lines)


class WebClueFinder:
    def find(self, query: str, max_items: int = 5) -> List[Dict[str, str]]:
        try:
            results: List[Dict[str, Any]] = []
            for row in DDGS().text(query, max_results=max_items):
                title = str(row.get("title", "")).strip()
                href = str(row.get("href", "")).strip()
                body = str(row.get("body", "")).strip()
                if not href:
                    continue
                alive = self._is_alive(href)
                tscore = trust_score(href)
                if not alive:
                    tscore -= 0.2
                results.append(
                    {
                        "title": title,
                        "url": href,
                        "snippet": body,
                        "trust": max(0.0, min(1.0, tscore)),
                        "alive": alive,
                    }
                )
            results.sort(key=lambda x: float(x.get("trust", 0.0)), reverse=True)
            top = results[:max_items]
            return [
                {
                    "title": str(x.get("title", "")),
                    "url": str(x.get("url", "")),
                    "snippet": str(x.get("snippet", "")),
                    "trust": f"{float(x.get('trust', 0.0)):.2f}",
                    "alive": "yes" if bool(x.get("alive", False)) else "no",
                }
                for x in top
            ]
        except Exception as exc:
            logger.warning("Web clue search failed: %s", exc)
            return []

    def _is_alive(self, url: str) -> bool:
        try:
            resp = requests.head(url, timeout=5, allow_redirects=True)
            return 200 <= int(resp.status_code) < 400
        except Exception:
            return False


embedder = Embedder()
store = MemoryStore(DATA_FILE, embedder)
code_assistant = CodeAssistant()
news_watcher = NewsWatcher(store)
web_clue_finder = WebClueFinder()


def user_key_from_update(update: Update) -> str:
    if update.effective_user and update.effective_user.id is not None:
        return f"user:{update.effective_user.id}"
    if update.effective_chat and update.effective_chat.id is not None:
        return f"chat:{update.effective_chat.id}"
    return "global"


def is_admin_user(update: Update) -> bool:
    user = update.effective_user
    if not user:
        return False
    return store.is_admin(user.id)


def user_id(update: Update) -> int:
    return int(update.effective_user.id) if update.effective_user and update.effective_user.id else 0


def check_rate_limit(uid: int) -> bool:
    if uid <= 0:
        return True
    now = datetime.now(tz=timezone.utc).timestamp()
    rows = RATE_LIMIT_STATE.get(uid, [])
    rows = [x for x in rows if now - x <= RATE_LIMIT_WINDOW_SEC]
    if len(rows) >= RATE_LIMIT_MAX_ACTIONS:
        RATE_LIMIT_STATE[uid] = rows
        return False
    rows.append(now)
    RATE_LIMIT_STATE[uid] = rows
    return True


def is_moderation_safe_text(text: str) -> bool:
    lower = text.lower()
    if "http://" in lower or "https://" in lower:
        return True
    suspicious = ["free crack", "keygen", "payload", "stealer", "ransomware builder"]
    return not any(x in lower for x in suspicious)


def channel_key_from_chat(chat: Any) -> str:
    chat_id = getattr(chat, "id", None)
    username = getattr(chat, "username", None)
    if username:
        return f"@{username.lower()}"
    return f"id:{chat_id}"


def channel_link(chat: Any, message_id: int) -> str:
    username = getattr(chat, "username", None)
    if username:
        return f"https://t.me/{username}/{message_id}"
    return f"(private channel message_id={message_id})"


def command_from_text(text: str) -> str:
    token = (text or "").strip().split(" ")[0].strip()
    if not token.startswith("/"):
        return ""
    # Support channel style commands like /ping@BotUsername
    if "@" in token:
        token = token.split("@", 1)[0]
    return token.lower()


def extract_file_meta(msg: Any) -> Dict[str, Any]:
    text = (msg.caption or msg.text or "").strip()
    file_name = ""
    mime_type = ""
    file_size = 0
    kind = "text"
    file_id = ""
    file_unique_id = ""
    if msg.document:
        kind = "document"
        file_name = msg.document.file_name or ""
        mime_type = msg.document.mime_type or ""
        file_size = int(msg.document.file_size or 0)
        file_id = msg.document.file_id or ""
        file_unique_id = msg.document.file_unique_id or ""
    elif msg.video:
        kind = "video"
        file_name = msg.video.file_name or "video"
        mime_type = msg.video.mime_type or ""
        file_size = int(msg.video.file_size or 0)
        file_id = msg.video.file_id or ""
        file_unique_id = msg.video.file_unique_id or ""
    elif msg.audio:
        kind = "audio"
        file_name = msg.audio.file_name or "audio"
        mime_type = msg.audio.mime_type or ""
        file_size = int(msg.audio.file_size or 0)
        file_id = msg.audio.file_id or ""
        file_unique_id = msg.audio.file_unique_id or ""
    elif msg.photo:
        kind = "photo"
        file_name = "photo"
        mime_type = "image/jpeg"
        file_size = int(msg.photo[-1].file_size or 0) if msg.photo else 0
        if msg.photo:
            file_id = msg.photo[-1].file_id or ""
            file_unique_id = msg.photo[-1].file_unique_id or ""
    elif msg.text:
        kind = "text"

    tags = re.findall(r"#([A-Za-z0-9_]+)", text)
    ext = Path(file_name.lower()).suffix if file_name else ""

    search_text = " | ".join(
        [
            kind,
            file_name,
            mime_type,
            ext,
            " ".join(tags),
            text,
        ]
    ).strip()
    return {
        "kind": kind,
        "file_name": file_name,
        "mime_type": mime_type,
        "file_size": file_size,
        "caption_or_text": text,
        "search_text": search_text,
        "tags": tags,
        "ext": ext,
        "file_id": file_id,
        "file_unique_id": file_unique_id,
        "is_safe_ext": is_safe_extension(file_name or ""),
    }


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    if update.effective_chat:
        store.subscribe_news(update.effective_chat.id)
    msg = (
        f"Hi, I am {PERSONA_NAME}.\n"
        "I adapt my personality to your conversation style.\n"
        "You are auto-subscribed to cyber/CVE updates. Use /help and start chatting."
    )
    await update.message.reply_text(persona_reply(msg, profile))


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("pong")


async def channel_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat:
        await update.message.reply_text("No chat context.")
        return
    ch_key = channel_key_from_chat(chat)
    count = store.channel_item_count(ch_key)
    lines = [
        "Channel status:",
        f"- chat_type: {chat.type}",
        f"- channel_key: {ch_key}",
        f"- indexed_items: {count}",
        f"- index_ack: {'on' if CHANNEL_INDEX_ACK else 'off'}",
    ]
    await update.message.reply_text("\n".join(lines))


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "Commands:\n"
        "/learn <key> = <value>  -> store a fact\n"
        "/ping                   -> quick bot liveness check\n"
        "/channel_status         -> channel indexing status\n"
        "/forget <key>           -> remove a fact\n"
        "/stats                  -> memory stats\n"
        "/retrain                -> rebuild all embeddings\n"
        "/readcode <path>        -> read file with line numbers\n"
        "/findfile <query>       -> find files/posts indexed from channels\n"
        "   filters: type: from: date:YYYY-MM-DD size:>1000000 safe:true page:2\n"
        "/indexreply             -> index a replied message/file (admin)\n"
        "/postchannel <text>     -> post a message to target channel (admin)\n"
        "/postfile               -> reply to a message and post it to channel (admin)\n"
        "/schedulepost YYYY-MM-DD HH:MM | text (UTC)\n"
        "/listschedule           -> list scheduled posts\n"
        "/cancelschedule <id>    -> cancel scheduled post (admin)\n"
        "/subscribe_news         -> start cyber/CVE push updates\n"
        "/unsubscribe_news       -> stop push updates\n"
        "/newsnow                -> fetch latest cyber digest now\n"
        "/newsstatus             -> show news push status\n"
        "/newsource              -> list all news sources\n"
        "/autonomous_on          -> enable autonomous mode\n"
        "/autonomous_off         -> disable autonomous mode\n"
        "/autonomous_status      -> autonomous mode status\n"
        "/admin_add <user_id>    -> add admin\n"
        "/admin_remove <user_id> -> remove admin\n"
        "/admin_list             -> list admins\n"
        "/topqueries             -> top searched queries\n"
        "/topfiles               -> most indexed files\n"
        "/missing_requests       -> open missing file requests\n"
        "/weekly_report          -> weekly analytics summary\n"
        "/healthcheck            -> verify bot runtime configuration\n"
        "/backupdata             -> backup state file (admin)\n"
        "/restoredata <path>     -> restore state file (admin)\n"
        "\nChat normally and I will adapt my tone based on your style."
    )
    await update.message.reply_text(msg)


async def learn(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    text = update.message.text.replace("/learn", "", 1).strip()
    if "=" not in text:
        await update.message.reply_text(persona_reply("Format: /learn key = value", profile))
        return

    key, value = [part.strip() for part in text.split("=", 1)]
    if not key or not value:
        await update.message.reply_text(
            persona_reply("Both key and value are required.", profile)
        )
        return

    store.update_profile_from_text(user_key, text)
    store.learn_fact(key, value)
    profile = store.get_adaptive_profile(user_key)
    await update.message.reply_text(persona_reply(f"Stored: {key} -> {value}", profile))


async def forget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    key = update.message.text.replace("/forget", "", 1).strip()
    if not key:
        await update.message.reply_text(persona_reply("Format: /forget key", profile))
        return

    store.update_profile_from_text(user_key, key)
    profile = store.get_adaptive_profile(user_key)
    if store.forget_fact(key):
        await update.message.reply_text(persona_reply(f"Deleted from memory: {key}", profile))
    else:
        await update.message.reply_text(persona_reply("I do not have that fact.", profile))


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    await update.message.reply_text(persona_reply(store.stats(), profile))


async def retrain(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    counts = store.retrain_embeddings()
    await update.message.reply_text(
        persona_reply(
            "Retraining complete.\n"
            f"Facts re-embedded: {counts['facts']}\n"
            f"Samples re-embedded: {counts['samples']}\n"
            f"Backend: {store.embedder.backend}",
            profile,
        )
    )


async def readcode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    args = context.args
    if not args:
        await update.message.reply_text(persona_reply("Format: /readcode <relative_path>", profile))
        return
    rel_path = " ".join(args).strip()
    result = code_assistant.read_code(rel_path)
    text = result.text
    if len(text) > MAX_TELEGRAM_REPLY_CHARS:
        text = text[:MAX_TELEGRAM_REPLY_CHARS] + "\n... [truncated]"
    await update.message.reply_text(persona_reply(text, profile))


async def index_channel_post(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    chat = update.effective_chat
    if not msg or not chat:
        return
    if getattr(chat, "type", "") != "channel":
        return

    cmd = command_from_text(msg.text or "")
    if cmd == "/ping":
        try:
            await context.bot.send_message(chat_id=chat.id, text="pong")
        except Exception as exc:
            logger.warning("Channel /ping response failed: %s", exc)
        return
    if cmd == "/channel_status":
        ch_key = channel_key_from_chat(chat)
        count = store.channel_item_count(ch_key)
        lines = [
            "Channel status:",
            f"- chat_type: {chat.type}",
            f"- channel_key: {ch_key}",
            f"- indexed_items: {count}",
            f"- index_ack: {'on' if CHANNEL_INDEX_ACK else 'off'}",
        ]
        try:
            await context.bot.send_message(chat_id=chat.id, text="\n".join(lines))
        except Exception as exc:
            logger.warning("Channel /channel_status response failed: %s", exc)
        return

    meta = extract_file_meta(msg)
    channel_key = channel_key_from_chat(chat)
    item = {
        "channel_key": channel_key,
        "channel_id": chat.id,
        "channel_username": chat.username or "",
        "channel_title": chat.title or "",
        "message_id": msg.message_id,
        "kind": meta["kind"],
        "file_name": meta["file_name"],
        "mime_type": meta["mime_type"],
        "file_size": meta["file_size"],
        "file_id": meta["file_id"],
        "file_unique_id": meta["file_unique_id"],
        "is_safe_ext": meta["is_safe_ext"],
        "tags": meta["tags"],
        "ext": meta["ext"],
        "caption_or_text": meta["caption_or_text"],
        "search_text": meta["search_text"],
        "embedding": embedder.embed(meta["search_text"]),
        "date": (msg.date or datetime.now(tz=timezone.utc)).astimezone(timezone.utc).isoformat(),
        "link": channel_link(chat, msg.message_id),
    }
    if item["file_name"] and not item["is_safe_ext"]:
        logger.warning("Unsafe extension indexed in channel: %s", item["file_name"])
    store.upsert_channel_item(channel_key, item)
    if CHANNEL_INDEX_ACK:
        try:
            await context.bot.send_message(
                chat_id=chat.id,
                text=(
                    f"Indexed: {item.get('file_name') or item.get('kind')} "
                    f"(total indexed: {store.channel_item_count(channel_key)})"
                ),
            )
        except Exception as exc:
            logger.warning("Channel ACK failed: %s", exc)
    fulfilled = store.fulfill_missing_requests(item["search_text"], item["link"])
    for req in fulfilled[:10]:
        try:
            await context.bot.send_message(
                chat_id=int(req["chat_id"]),
                text=(
                    f"Requested file clue matched for query: {req['query']}\n"
                    f"Found in channel: {item.get('channel_title') or item.get('channel_key')}\n"
                    f"Link: {item['link']}"
                ),
                disable_web_page_preview=True,
            )
        except Exception as exc:
            logger.warning("Missing-request notify failed: %s", exc)


async def indexreply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    if not await require_admin(update, profile):
        return
    msg = update.effective_message
    if not msg or not msg.reply_to_message:
        await update.message.reply_text(persona_reply("Reply to a message and run /indexreply", profile))
        return
    src = msg.reply_to_message
    chat = src.chat
    meta = extract_file_meta(src)
    ch_key = channel_key_from_chat(chat)
    item = {
        "channel_key": ch_key,
        "channel_id": chat.id,
        "channel_username": getattr(chat, "username", "") or "",
        "channel_title": getattr(chat, "title", "") or "",
        "message_id": src.message_id,
        "kind": meta["kind"],
        "file_name": meta["file_name"],
        "mime_type": meta["mime_type"],
        "file_size": meta["file_size"],
        "file_id": meta["file_id"],
        "file_unique_id": meta["file_unique_id"],
        "is_safe_ext": meta["is_safe_ext"],
        "tags": meta["tags"],
        "ext": meta["ext"],
        "caption_or_text": meta["caption_or_text"],
        "search_text": meta["search_text"],
        "embedding": embedder.embed(meta["search_text"]),
        "date": (src.date or datetime.now(tz=timezone.utc)).astimezone(timezone.utc).isoformat(),
        "link": channel_link(chat, src.message_id),
    }
    store.upsert_channel_item(ch_key, item)
    await update.message.reply_text(persona_reply("Replied message indexed.", profile))


def format_found_items(items: List[Dict[str, Any]], title: str) -> str:
    lines = [title]
    for row in items:
        size = int(row.get("file_size", 0))
        size_text = f"{size} bytes" if size > 0 else "size n/a"
        lines.append(
            f"- [{row.get('kind', 'file')}] {row.get('file_name', '(no file name)')}\n"
            f"  Channel: {row.get('channel_title') or row.get('channel_key')}\n"
            f"  Mime: {row.get('mime_type') or 'n/a'} | {size_text}\n"
            f"  Score: {float(row.get('score', 0.0)):.2f}\n"
            f"  Link: {row.get('link')}"
        )
    return "\n".join(lines)


def format_web_clues(clues: List[Dict[str, str]], query: str) -> str:
    if not clues:
        return (
            f"No indexed file found for '{query}', and no web clues were available right now."
        )
    lines = [f"No indexed file found for '{query}'. Web clues:"]
    for row in clues:
        title = row.get("title") or "Untitled"
        url = row.get("url") or ""
        snippet = row.get("snippet") or ""
        trust = row.get("trust", "0.00")
        alive = row.get("alive", "no")
        lines.append(f"- {title}\n  {url}\n  trust={trust} alive={alive}\n  {snippet}")
    return "\n".join(lines)


def parse_findfile_args(args: List[str]) -> tuple[str, Dict[str, str], int]:
    filters: Dict[str, str] = {}
    terms: List[str] = []
    page = 1
    for token in args:
        if ":" in token:
            k, v = token.split(":", 1)
            key = k.strip().lower()
            val = v.strip()
            if key in {"type", "from", "date", "size", "safe"} and val:
                filters[key] = val
                continue
            if key == "page" and val.isdigit():
                page = max(1, int(val))
                continue
        terms.append(token)
    return " ".join(terms).strip(), filters, page


def make_find_session(
    query: str,
    filters: Dict[str, str],
    user_id_value: int,
    chat_id_value: int,
) -> str:
    sid = secrets.token_hex(4)
    FIND_SESSIONS[sid] = {
        "query": query,
        "filters": dict(filters),
        "user_id": int(user_id_value),
        "chat_id": int(chat_id_value),
        "created_at": datetime.now(tz=timezone.utc).timestamp(),
    }
    # Keep in-memory state bounded.
    if len(FIND_SESSIONS) > 2000:
        stale_keys = sorted(
            FIND_SESSIONS.keys(),
            key=lambda k: float(FIND_SESSIONS[k].get("created_at", 0.0)),
        )[:400]
        for key in stale_keys:
            FIND_SESSIONS.pop(key, None)
    return sid


def get_find_session(sid: str) -> Dict[str, Any] | None:
    sess = FIND_SESSIONS.get(sid)
    if not sess:
        return None
    age = datetime.now(tz=timezone.utc).timestamp() - float(sess.get("created_at", 0.0))
    if age > 3600:
        FIND_SESSIONS.pop(sid, None)
        return None
    return sess


async def require_admin(update: Update, profile: Dict[str, Any]) -> bool:
    if is_admin_user(update):
        return True
    uid = user_id(update)
    if uid:
        store.record_abuse(uid, "admin_command_denied")
    await update.message.reply_text(persona_reply("Admin permission required.", profile))
    return False


async def findfile(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    uid = user_id(update)
    if not check_rate_limit(uid):
        store.record_abuse(uid, "rate_limited_findfile")
        await update.message.reply_text(persona_reply("Rate limit exceeded. Try again shortly.", profile))
        return
    query, parsed_filters, page = parse_findfile_args(context.args)
    if not query:
        await update.message.reply_text(persona_reply("Format: /findfile <query>", profile))
        return
    sid = make_find_session(
        query=query,
        filters=parsed_filters,
        user_id_value=user_id(update),
        chat_id_value=int(update.effective_chat.id) if update.effective_chat else 0,
    )
    text, markup, found = render_findfile_page(sid, page)
    if update.effective_chat and update.effective_user:
        store.record_query(update.effective_chat.id, update.effective_user.id, query, found)
    if found == 0 and update.effective_chat and update.effective_user:
        store.add_missing_request(update.effective_chat.id, update.effective_user.id, query)
    await update.message.reply_text(text, reply_markup=markup, disable_web_page_preview=True)


def render_findfile_page(sid: str, page: int) -> tuple[str, InlineKeyboardMarkup | None, int]:
    sess = get_find_session(sid)
    if not sess:
        return ("Search session expired. Run /findfile again.", None, 0)
    query = str(sess.get("query", ""))
    filters = dict(sess.get("filters", {}))
    page = max(1, int(page))
    offset = (page - 1) * 6
    matches = store.search_channel_items(query, limit=6, filters=filters, offset=offset)
    if matches:
        out = format_found_items(matches, f"Found {len(matches)} matching items (page {page}):")
        if len(out) > MAX_TELEGRAM_REPLY_CHARS:
            out = out[:MAX_TELEGRAM_REPLY_CHARS] + "\n... [truncated]"
        buttons = []
        if page > 1:
            buttons.append(InlineKeyboardButton("Prev", callback_data=f"ff:{sid}:{page-1}"))
        buttons.append(InlineKeyboardButton("Refresh", callback_data=f"ff:{sid}:{page}"))
        buttons.append(InlineKeyboardButton("Next", callback_data=f"ff:{sid}:{page+1}"))
        return out, InlineKeyboardMarkup([buttons]), len(matches)

    clues = web_clue_finder.find(
        f"{query} file download telegram channel cybersecurity resources",
        max_items=5,
    )
    out = format_web_clues(clues, query)
    if len(out) > MAX_TELEGRAM_REPLY_CHARS:
        out = out[:MAX_TELEGRAM_REPLY_CHARS] + "\n... [truncated]"
    return out, None, 0


async def findfile_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q or not q.data:
        return
    if not q.data.startswith("ff:"):
        return
    parts = q.data.split(":")
    if len(parts) != 3:
        await q.answer("Invalid pagination token", show_alert=True)
        return
    sid = parts[1]
    try:
        page = int(parts[2])
    except Exception:
        await q.answer("Invalid page", show_alert=True)
        return
    sess = get_find_session(sid)
    if not sess:
        await q.answer("Session expired", show_alert=True)
        return
    uid = user_id(update)
    if uid and int(sess.get("user_id", 0)) not in {0, uid}:
        await q.answer("Not your search session", show_alert=True)
        return
    text, markup, _ = render_findfile_page(sid, page)
    try:
        await q.edit_message_text(text=text, reply_markup=markup, disable_web_page_preview=True)
    except Exception:
        await q.message.reply_text(text, reply_markup=markup, disable_web_page_preview=True)
    await q.answer()


async def healthcheck(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    checks: List[str] = []
    ok = 0
    total = 0

    def add_check(name: str, passed: bool, detail: str) -> None:
        nonlocal ok, total
        total += 1
        if passed:
            ok += 1
            checks.append(f"[OK] {name}: {detail}")
        else:
            checks.append(f"[FAIL] {name}: {detail}")

    add_check("BOT_TOKEN", bool(os.getenv("BOT_TOKEN")), "configured" if os.getenv("BOT_TOKEN") else "missing")
    add_check("JobQueue", bool(context.job_queue), "available" if context.job_queue else "unavailable")
    add_check("Embedding", True, store.embedder.backend)
    add_check("TargetChannel", bool(TARGET_CHANNEL_ID), TARGET_CHANNEL_ID or "not configured")
    add_check("Admins", len(store.admin_ids()) > 0 or len(ADMIN_USER_IDS) > 0, f"{len(store.admin_ids())} persisted")

    # Write test
    try:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        test_file = BACKUP_DIR / ".health.tmp"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        add_check("Filesystem", True, str(BACKUP_DIR))
    except Exception as exc:
        add_check("Filesystem", False, str(exc))

    # Internet/feed test
    try:
        first_feed = NEWS_FEEDS[0][1]
        resp = requests.get(first_feed, timeout=8)
        add_check("InternetFeed", 200 <= resp.status_code < 400, f"status={resp.status_code}")
    except Exception as exc:
        add_check("InternetFeed", False, str(exc))

    # Bot channel permission check if configured
    if TARGET_CHANNEL_ID:
        try:
            me = await context.bot.get_me()
            member = await context.bot.get_chat_member(chat_id=TARGET_CHANNEL_ID, user_id=me.id)
            status = str(getattr(member, "status", "unknown"))
            add_check("ChannelAccess", status in {"administrator", "member", "creator"}, status)
        except Exception as exc:
            add_check("ChannelAccess", False, str(exc))
    else:
        add_check("ChannelAccess", False, "TARGET_CHANNEL_ID not set")

    summary = f"Healthcheck: {ok}/{total} passed\n" + "\n".join(checks)
    await update.message.reply_text(persona_reply(summary, profile))


async def postchannel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    uid = user_id(update)
    if not check_rate_limit(uid):
        store.record_abuse(uid, "rate_limited_postchannel")
        await update.message.reply_text(persona_reply("Rate limit exceeded. Try again shortly.", profile))
        return
    if not await require_admin(update, profile):
        return
    msg = " ".join(context.args).strip()
    if not msg:
        await update.message.reply_text(
            persona_reply("Format: /postchannel <text>", profile)
        )
        return
    if not is_moderation_safe_text(msg):
        store.record_abuse(uid, "postchannel_blocked_moderation")
        await update.message.reply_text(persona_reply("Blocked by moderation policy.", profile))
        return

    channel_id = TARGET_CHANNEL_ID
    if not channel_id:
        await update.message.reply_text(
            persona_reply("TARGET_CHANNEL_ID is not configured in .env.", profile)
        )
        return
    try:
        sent = await context.bot.send_message(chat_id=channel_id, text=msg)
        await update.message.reply_text(
            persona_reply(f"Posted to channel successfully (message id: {sent.message_id}).", profile)
        )
    except Exception as exc:
        await update.message.reply_text(
            persona_reply(f"Failed to post to channel: {exc}", profile)
        )


async def postfile(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    uid = user_id(update)
    if not await require_admin(update, profile):
        return
    if not TARGET_CHANNEL_ID:
        await update.message.reply_text(persona_reply("TARGET_CHANNEL_ID is not configured.", profile))
        return
    msg = update.effective_message
    if not msg or not msg.reply_to_message:
        await update.message.reply_text(persona_reply("Reply to a file/message and run /postfile", profile))
        return
    src = msg.reply_to_message
    meta = extract_file_meta(src)
    if meta["file_name"] and not meta["is_safe_ext"]:
        store.record_abuse(uid, "postfile_blocked_unsafe_ext")
        await update.message.reply_text(persona_reply("Blocked unsafe file extension.", profile))
        return
    try:
        sent = await context.bot.copy_message(
            chat_id=TARGET_CHANNEL_ID,
            from_chat_id=src.chat_id,
            message_id=src.message_id,
        )
        await update.message.reply_text(persona_reply(f"Posted file/message to channel. id={sent.message_id}", profile))
    except Exception as exc:
        await update.message.reply_text(persona_reply(f"Failed to post file: {exc}", profile))


def parse_schedule_payload(raw: str) -> tuple[datetime | None, str]:
    if "|" not in raw:
        return None, ""
    left, right = raw.split("|", 1)
    ts = left.strip()
    txt = right.strip()
    try:
        when = datetime.strptime(ts, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    except Exception:
        return None, ""
    return when, txt


async def schedulepost(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    if not await require_admin(update, profile):
        return
    if not TARGET_CHANNEL_ID:
        await update.message.reply_text(persona_reply("TARGET_CHANNEL_ID is not configured.", profile))
        return
    raw = (update.effective_message.text or "").replace("/schedulepost", "", 1).strip()
    when, text = parse_schedule_payload(raw)
    if not when or not text:
        await update.message.reply_text(
            persona_reply("Format: /schedulepost YYYY-MM-DD HH:MM | your text (UTC)", profile)
        )
        return
    post_id = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    row = {
        "id": post_id,
        "channel_id": TARGET_CHANNEL_ID,
        "text": text[:3000],
        "when": when.isoformat(),
        "created_by": user_id(update),
    }
    store.add_scheduled_post(row)
    if context.job_queue:
        delay = max(1, int((when - datetime.now(tz=timezone.utc)).total_seconds()))
        context.job_queue.run_once(send_scheduled_post_job, when=delay, data=row, name=f"sched-{post_id}")
    await update.message.reply_text(persona_reply(f"Scheduled post created. id={post_id}", profile))


async def listschedule(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    rows = store.list_scheduled_posts()
    if not rows:
        await update.message.reply_text(persona_reply("No scheduled posts.", profile))
        return
    lines = ["Scheduled posts:"]
    for row in rows[:20]:
        lines.append(f"- id={row.get('id')} when={row.get('when')} text={str(row.get('text',''))[:80]}")
    await update.message.reply_text(persona_reply("\n".join(lines), profile))


async def cancelschedule(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    if not await require_admin(update, profile):
        return
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text(persona_reply("Format: /cancelschedule <id>", profile))
        return
    post_id = int(context.args[0])
    removed = store.remove_scheduled_post(post_id)
    await update.message.reply_text(persona_reply("Canceled." if removed else "Schedule id not found.", profile))


async def admin_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    if not await require_admin(update, profile):
        return
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text(persona_reply("Format: /admin_add <user_id>", profile))
        return
    target = int(context.args[0])
    ok = store.add_admin(target)
    await update.message.reply_text(persona_reply("Admin added." if ok else "Already admin.", profile))


async def admin_remove(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    if not await require_admin(update, profile):
        return
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text(persona_reply("Format: /admin_remove <user_id>", profile))
        return
    target = int(context.args[0])
    ok = store.remove_admin(target)
    await update.message.reply_text(persona_reply("Admin removed." if ok else "Not an admin.", profile))


async def admin_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    admins = store.admin_ids()
    await update.message.reply_text(persona_reply("Admins: " + ", ".join(str(x) for x in admins), profile))


async def topqueries(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    rows = store.top_queries(days=7, limit=10)
    if not rows:
        await update.message.reply_text(persona_reply("No query analytics yet.", profile))
        return
    lines = ["Top queries (7d):"]
    for q, c in rows:
        lines.append(f"- {q} ({c})")
    await update.message.reply_text(persona_reply("\n".join(lines), profile))


async def topfiles(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    rows = store.top_files(limit=10)
    if not rows:
        await update.message.reply_text(persona_reply("No files indexed yet.", profile))
        return
    lines = ["Top files:"]
    for name, c in rows:
        lines.append(f"- {name} ({c})")
    await update.message.reply_text(persona_reply("\n".join(lines), profile))


async def missing_requests_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    rows = store.open_missing_requests()
    if not rows:
        await update.message.reply_text(persona_reply("No open missing requests.", profile))
        return
    lines = ["Open missing requests:"]
    for row in rows[:20]:
        lines.append(f"- id={row.get('id')} query={row.get('query')} chat={row.get('chat_id')}")
    await update.message.reply_text(persona_reply("\n".join(lines), profile))


async def weekly_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    top_q = store.top_queries(days=7, limit=5)
    top_f = store.top_files(limit=5)
    open_miss = len(store.open_missing_requests())
    lines = ["Weekly report:", f"- Open missing requests: {open_miss}", "- Top queries:"]
    for q, c in top_q:
        lines.append(f"  {q} ({c})")
    lines.append("- Top files:")
    for n, c in top_f:
        lines.append(f"  {n} ({c})")
    await update.message.reply_text(persona_reply("\n".join(lines), profile))


async def backupdata(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    if not await require_admin(update, profile):
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = BACKUP_DIR / f"knowledge_{stamp}.json"
    shutil.copyfile(DATA_FILE, out)
    await update.message.reply_text(persona_reply(f"Backup created: {out}", profile))


async def restoredata(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    if not await require_admin(update, profile):
        return
    if not context.args:
        await update.message.reply_text(persona_reply("Format: /restoredata <backup_file_path>", profile))
        return
    src = Path(" ".join(context.args).strip())
    if not src.exists() or not src.is_file():
        await update.message.reply_text(persona_reply("Backup file not found.", profile))
        return
    shutil.copyfile(src, DATA_FILE)
    await update.message.reply_text(persona_reply("Restore complete. Restart bot to fully reload state.", profile))


async def subscribe_news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    if not update.effective_chat:
        await update.message.reply_text(persona_reply("Unable to resolve chat id.", profile))
        return
    added = store.subscribe_news(update.effective_chat.id)
    if added:
        text = (
            "Subscribed to cybersecurity news. "
            f"Updates will be pushed every {NEWS_INTERVAL_MINUTES} minutes."
        )
    else:
        text = "This chat is already subscribed."
    await update.message.reply_text(persona_reply(text, profile))


async def unsubscribe_news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    if not update.effective_chat:
        await update.message.reply_text(persona_reply("Unable to resolve chat id.", profile))
        return
    removed = store.unsubscribe_news(update.effective_chat.id)
    if removed:
        text = "Unsubscribed from cybersecurity news updates."
    else:
        text = "This chat is not subscribed."
    await update.message.reply_text(persona_reply(text, profile))


async def newsnow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    items = await news_watcher.fetch_latest()
    digest = news_watcher.format_digest(items[:MAX_NEWS_ITEMS_PER_PUSH], "Cybersecurity Digest")
    if len(digest) > MAX_TELEGRAM_REPLY_CHARS:
        digest = digest[:MAX_TELEGRAM_REPLY_CHARS] + "\n... [truncated]"
    await update.message.reply_text(digest, disable_web_page_preview=True)


async def newsstatus(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    if not update.effective_chat:
        await update.message.reply_text(persona_reply("Unable to resolve chat id.", profile))
        return
    chat_id = update.effective_chat.id
    subscribed = chat_id in store.news_subscribers()
    text = (
        f"News interval: {NEWS_INTERVAL_MINUTES} minutes\n"
        f"Autonomous mode: {'on' if store.is_autonomous_enabled() else 'off'}\n"
        f"Subscribed in this chat: {'yes' if subscribed else 'no'}\n"
        f"Total subscribed chats: {len(store.news_subscribers())}"
    )
    await update.message.reply_text(persona_reply(text, profile))


async def newsource(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    lines = ["News Sources:"]
    for idx, (name, url) in enumerate(NEWS_FEEDS, start=1):
        lines.append(f"{idx}. {name}\n   {url}")
    lines.append("These are the feeds used for cybersecurity/CVE updates.")
    text = "\n".join(lines)
    await update.message.reply_text(persona_reply(text, profile), disable_web_page_preview=True)


async def autonomous_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    store.set_autonomous_enabled(True)
    await update.message.reply_text(
        persona_reply(
            "Autonomous mode enabled. Monitoring and proactive briefs are active.",
            profile,
        )
    )


async def autonomous_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    store.set_autonomous_enabled(False)
    await update.message.reply_text(
        persona_reply(
            "Autonomous mode disabled. Background monitoring is paused.",
            profile,
        )
    )


async def autonomous_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_key = user_key_from_update(update)
    profile = store.get_adaptive_profile(user_key)
    text = (
        f"Autonomous mode: {'on' if store.is_autonomous_enabled() else 'off'}\n"
        f"News poll interval: {NEWS_INTERVAL_MINUTES} minutes\n"
        f"Trend brief interval: {AUTONOMOUS_BRIEF_HOURS} hours"
    )
    await update.message.reply_text(persona_reply(text, profile))


async def push_news_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not store.is_autonomous_enabled():
        return
    subscribers = store.news_subscribers()
    if not subscribers:
        return
    fresh = await news_watcher.fetch_fresh(MAX_NEWS_ITEMS_PER_PUSH)
    if not fresh:
        return
    digest = news_watcher.format_digest(fresh, "Cybersecurity Update")
    if len(digest) > MAX_TELEGRAM_REPLY_CHARS:
        digest = digest[:MAX_TELEGRAM_REPLY_CHARS] + "\n... [truncated]"
    for chat_id in subscribers:
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=digest,
                disable_web_page_preview=True,
            )
        except Exception as exc:
            logger.warning("Failed to send news update to %s: %s", chat_id, exc)
            store.push_retry({"type": "send_message", "chat_id": chat_id, "text": digest})


async def autonomous_trend_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not store.is_autonomous_enabled():
        return
    subscribers = store.news_subscribers()
    if not subscribers:
        return
    items = await news_watcher.fetch_latest()
    brief = news_watcher.format_trend_brief(items[:15])
    if len(brief) > MAX_TELEGRAM_REPLY_CHARS:
        brief = brief[:MAX_TELEGRAM_REPLY_CHARS] + "\n... [truncated]"
    for chat_id in subscribers:
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=brief,
                disable_web_page_preview=True,
            )
        except Exception as exc:
            logger.warning("Failed to send autonomous brief to %s: %s", chat_id, exc)
            store.push_retry({"type": "send_message", "chat_id": chat_id, "text": brief})


async def send_scheduled_post_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    row = context.job.data if context.job else None
    if not row:
        return
    channel_id = row.get("channel_id")
    text = str(row.get("text", ""))
    post_id = int(row.get("id", 0))
    try:
        await context.bot.send_message(chat_id=channel_id, text=text)
    except Exception as exc:
        logger.warning("Scheduled post failed (id=%s): %s", post_id, exc)
        store.push_retry({"type": "send_message", "chat_id": channel_id, "text": text})
    finally:
        store.remove_scheduled_post(post_id)


async def retry_queue_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    batch = store.pop_retry_batch(limit=20)
    if not batch:
        return
    failed: List[Dict[str, Any]] = []
    for row in batch:
        typ = str(row.get("type", ""))
        if typ != "send_message":
            continue
        try:
            await context.bot.send_message(
                chat_id=row.get("chat_id"),
                text=str(row.get("text", ""))[:MAX_TELEGRAM_REPLY_CHARS],
                disable_web_page_preview=True,
            )
        except Exception:
            failed.append(row)
    for row in failed:
        store.push_retry(row)


async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err = str(getattr(context, "error", "unknown error"))
    logger.exception("Unhandled error: %s", err)
    if ALERT_CHAT_ID:
        try:
            await context.bot.send_message(
                chat_id=ALERT_CHAT_ID,
                text=f"[ALERT] Bot error: {err[:1500]}",
            )
        except Exception:
            logger.warning("Failed to deliver alert to ALERT_CHAT_ID")


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg:
        return
    user_text = (msg.text or "").strip()
    if not user_text:
        return
    if update.effective_chat and update.effective_chat.type == "channel":
        return
    user_key = user_key_from_update(update)
    store.update_profile_from_text(user_key, user_text)
    profile = store.get_adaptive_profile(user_key)
    lang = detect_language(user_text)

    direct_fact = store.get_fact(user_text)
    if direct_fact:
        response = persona_reply(f"Memory: {direct_fact}", profile)
        await update.message.reply_text(response)
        store.learn_sample(user_text, response)
        return

    fact_hits = store.semantic_facts(user_text)
    if fact_hits:
        response = persona_reply(
            "Memory links found:\n"
            + "\n".join(
                [
                    f"- {str(item['key'])}: {str(item['value'])} (score {float(item['score']):.2f})"
                    for item in fact_hits
                ]
            ),
            profile,
        )
        await update.message.reply_text(response)
        store.learn_sample(user_text, response)
        return

    similar = store.similar_responses(user_text)
    if similar:
        response = persona_reply(
            "Close to earlier chats:\n" + "\n".join([f"- {line}" for line in similar]),
            profile,
        )
    else:
        prompt_words = len(re.findall(r"\w+", user_text))
        if prompt_words <= 3:
            short_msg = (
                "Short ping detected. Give me more context, or teach me with /learn key = value."
                if lang == "en"
                else "à¦›à§‹à¦Ÿ à¦®à§‡à¦¸à§‡à¦œ à¦ªà§‡à§Ÿà§‡à¦›à¦¿à¥¤ à¦†à¦°à§‡à¦•à¦Ÿà§ à¦¬à¦¿à¦¸à§à¦¤à¦¾à¦°à¦¿à¦¤ à¦¬à¦²à§à¦¨, à¦…à¦¥à¦¬à¦¾ /learn key = value à¦¦à¦¿à§Ÿà§‡ à¦¶à§‡à¦–à¦¾à¦¨à¥¤"
            )
            response = persona_reply(
                short_msg,
                profile,
            )
        elif "who are you" in user_text.lower() or "your personality" in user_text.lower():
            who_msg = (
                f"I am {PERSONA_NAME}. I adapt to your conversation style in real time."
                if lang == "en"
                else f"à¦†à¦®à¦¿ {PERSONA_NAME}à¥¤ à¦†à¦®à¦¿ à¦†à¦ªà¦¨à¦¾à¦° à¦•à¦¥à§‹à¦ªà¦•à¦¥à¦¨à§‡à¦° à¦¸à§à¦Ÿà¦¾à¦‡à¦²à§‡ à¦¨à¦¿à¦œà§‡à¦•à§‡ à¦®à¦¾à¦¨à¦¿à§Ÿà§‡ à¦¨à§‡à¦‡à¥¤"
            )
            response = persona_reply(
                who_msg,
                profile,
            )
        else:
            learn_msg = (
                "I am learning this topic now. Teach me a fact with /learn key = value and I will remember it."
                if lang == "en"
                else "à¦†à¦®à¦¿ à¦à¦‡ à¦¬à¦¿à¦·à§Ÿà¦Ÿà¦¿ à¦¶à¦¿à¦–à¦›à¦¿à¥¤ /learn key = value à¦¦à¦¿à§Ÿà§‡ à¦¤à¦¥à§à¦¯ à¦¶à§‡à¦–à¦¾à¦¨, à¦†à¦®à¦¿ à¦®à¦¨à§‡ à¦°à¦¾à¦–à¦¬à¥¤"
            )
            response = persona_reply(
                learn_msg,
                profile,
            )

    await update.message.reply_text(response)
    store.learn_sample(user_text, response)


def main() -> None:
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN is missing. Put it in .env file.")

    # Python 3.14 no longer provides a default loop in main thread.
    # PTB 21.x still expects one when starting polling.
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("channel_status", channel_status))
    app.add_handler(CommandHandler("learn", learn))
    app.add_handler(CommandHandler("forget", forget))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("retrain", retrain))
    app.add_handler(CommandHandler("readcode", readcode))
    app.add_handler(CommandHandler("findfile", findfile))
    app.add_handler(CommandHandler("indexreply", indexreply))
    app.add_handler(CommandHandler("postchannel", postchannel))
    app.add_handler(CommandHandler("postfile", postfile))
    app.add_handler(CommandHandler("schedulepost", schedulepost))
    app.add_handler(CommandHandler("listschedule", listschedule))
    app.add_handler(CommandHandler("cancelschedule", cancelschedule))
    app.add_handler(CommandHandler("subscribe_news", subscribe_news))
    app.add_handler(CommandHandler("unsubscribe_news", unsubscribe_news))
    app.add_handler(CommandHandler("newsnow", newsnow))
    app.add_handler(CommandHandler("newsstatus", newsstatus))
    app.add_handler(CommandHandler("newsource", newsource))
    app.add_handler(CommandHandler("newsources", newsource))
    app.add_handler(MessageHandler(filters.ChatType.CHANNEL, index_channel_post))
    app.add_handler(CommandHandler("autonomous_on", autonomous_on))
    app.add_handler(CommandHandler("autonomous_off", autonomous_off))
    app.add_handler(CommandHandler("autonomous_status", autonomous_status))
    app.add_handler(CommandHandler("admin_add", admin_add))
    app.add_handler(CommandHandler("admin_remove", admin_remove))
    app.add_handler(CommandHandler("admin_list", admin_list))
    app.add_handler(CommandHandler("topqueries", topqueries))
    app.add_handler(CommandHandler("topfiles", topfiles))
    app.add_handler(CommandHandler("missing_requests", missing_requests_cmd))
    app.add_handler(CommandHandler("weekly_report", weekly_report))
    app.add_handler(CommandHandler("healthcheck", healthcheck))
    app.add_handler(CommandHandler("backupdata", backupdata))
    app.add_handler(CommandHandler("restoredata", restoredata))
    app.add_handler(CallbackQueryHandler(findfile_callback, pattern=r"^ff:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.add_error_handler(global_error_handler)

    if app.job_queue:
        # Re-register persisted scheduled posts.
        now = datetime.now(tz=timezone.utc)
        for row in store.list_scheduled_posts():
            try:
                when = datetime.fromisoformat(str(row.get("when")))
            except Exception:
                continue
            delay = int((when - now).total_seconds())
            if delay <= 0:
                delay = 3
            app.job_queue.run_once(
                send_scheduled_post_job,
                when=delay,
                data=row,
                name=f"sched-{row.get('id')}",
            )
        app.job_queue.run_repeating(
            push_news_job,
            interval=max(5, NEWS_INTERVAL_MINUTES) * 60,
            first=20,
            name="cyber-news-push",
        )
        app.job_queue.run_repeating(
            autonomous_trend_job,
            interval=max(1, AUTONOMOUS_BRIEF_HOURS) * 3600,
            first=120,
            name="autonomous-trend-brief",
        )
        app.job_queue.run_repeating(
            retry_queue_job,
            interval=60,
            first=30,
            name="retry-queue",
        )
    else:
        logger.warning("Job queue unavailable; automatic news push is disabled.")

    logger.info("Bot %s starting...", BOT_NAME)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

