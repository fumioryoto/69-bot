import os

os.environ["DISABLE_EMBED_MODEL"] = "1"

import bot  # noqa: E402


def test_parse_findfile_args():
    query, filters, page = bot.parse_findfile_args(
        ["linux", "iso", "type:document", "from:@mychan", "date:2026-02-11", "size:>1000", "page:3"]
    )
    assert query == "linux iso"
    assert filters["type"] == "document"
    assert filters["from"] == "@mychan"
    assert filters["date"] == "2026-02-11"
    assert filters["size"] == ">1000"
    assert page == 3


def test_detect_language():
    assert bot.detect_language("Hello world") == "en"
    assert bot.detect_language("হ্যালো") == "bn"


def test_trust_score_and_domain():
    assert bot.get_domain("https://nvd.nist.gov/vuln/detail/CVE-2024-1234") == "nvd.nist.gov"
    assert bot.trust_score("https://nvd.nist.gov/vuln/detail/CVE-2024-1234") >= 0.9
    assert bot.trust_score("https://example.org/file") <= 0.5


def test_safe_extension():
    assert bot.is_safe_extension("guide.pdf") is True
    assert bot.is_safe_extension("payload.exe") is False


def test_make_find_session_roundtrip():
    sid = bot.make_find_session("abc", {"type": "document"}, 10, 20)
    sess = bot.get_find_session(sid)
    assert sess is not None
    assert sess["query"] == "abc"
    assert sess["filters"]["type"] == "document"
