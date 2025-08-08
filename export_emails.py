import argparse
import base64
import datetime as dt
import email
import email.policy
import hashlib
import html as html_module
import imaplib
import json
import logging
import os
import re
import socket
import sys
import time
from dataclasses import dataclass
from email.header import decode_header, make_header
from email.message import EmailMessage, Message
from email.utils import getaddresses, parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from bs4 import BeautifulSoup  # type: ignore
    BS4_AVAILABLE = True
except Exception:
    BS4_AVAILABLE = False


# -----------------------------
# Utility helpers
# -----------------------------

def env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "t", "on"}


def to_bool(s: Optional[str], default: bool) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in {"1", "true", "yes", "y", "t", "on"}


def sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def decode_mime_words(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        decoded = str(make_header(decode_header(value)))
        return decoded
    except Exception:
        return value


def sanitize_filename(name: Optional[str]) -> str:
    if not name:
        return "attachment"
    # Decode MIME words if present
    name = decode_mime_words(name)
    # Remove path separators and control characters
    name = re.sub(r"[\/\\\:\*\?\"\<\>\|\r\n\t]+", "_", name)
    # Trim whitespace and dots
    name = name.strip().strip(".")
    if not name:
        name = "attachment"
    # Limit length
    if len(name) > 255:
        base, ext = os.path.splitext(name)
        name = (base[:200] + "_trunc") + ext[:50]
    return name


def parse_addresses(header_value: Optional[str]) -> List[Dict[str, str]]:
    if not header_value:
        return []
    addrs = []
    for name, email_addr in getaddresses([header_value]):
        name_dec = decode_mime_words(name).strip()
        email_norm = (email_addr or "").strip()
        addrs.append({"name": name_dec, "email": email_norm})
    return addrs


def parse_date_to_utc(date_header: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not date_header:
        return None, None
    try:
        dt_obj = parsedate_to_datetime(date_header)
        if dt_obj is None:
            return date_header, None
        if dt_obj.tzinfo is None:
            # Assume UTC if tz is missing
            dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
        dt_utc = dt_obj.astimezone(dt.timezone.utc)
        return date_header, dt_utc.isoformat().replace("+00:00", "Z")
    except Exception:
        return date_header, None


SUBJECT_PREFIXES = [
    "re", "fwd", "fw", "aw", "wg", "sv", "rv", "res", "enc", "vb", "tr",
    "ré", "antw", "antwort", "回复", "轉寄", "转发", "пере", "ответ", "回复", "轉發"
]
PREFIX_REGEX = re.compile(
    r"^\s*(?:(?:" + "|".join(re.escape(p) for p in SUBJECT_PREFIXES) + r")\s*(?:\[\d+\])?\s*:\s*)+",
    flags=re.IGNORECASE
)


def canonicalize_subject(subject: Optional[str]) -> str:
    if not subject:
        return ""
    s = decode_mime_words(subject)
    s = re.sub(r"\s+", " ", s).strip()
    # Remove duplicate/stacked prefixes like "Re:", "Fwd:", including localized variants
    prev = None
    while prev != s:
        prev = s
        s = re.sub(PREFIX_REGEX, "", s).strip()
    return s


def html_to_text(html: str) -> str:
    if not html:
        return ""
    if BS4_AVAILABLE:
        try:
            soup = BeautifulSoup(html, "html.parser")
            # Remove script/style
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            # Normalize whitespace
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"[ \t]{2,}", " ", text)
            return text.strip()
        except Exception:
            pass
    # Fallback: naive strip tags
    try:
        cleaned = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", html)
        cleaned = re.sub(r"(?s)<[^>]+>", " ", cleaned)
        cleaned = html_module.unescape(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()
    except Exception:
        return html


def extract_message_ids(value: Optional[str]) -> List[str]:
    if not value:
        return []
    # Extract tokens like <id@domain>
    ids = re.findall(r"<[^>]+>", value)
    # Normalize: lowercase, strip angle brackets and whitespace
    norm = [re.sub(r"^\s*<|>\s*$", "", i).strip().lower() for i in ids]
    return [i for i in norm if i]


def extract_single_message_id(value: Optional[str]) -> Optional[str]:
    ids = extract_message_ids(value)
    if not ids:
        return None
    # Some servers put only one; if multiple, take the first
    return ids[0]


def participants_set(from_addr: Dict[str, str], to_addrs: List[Dict[str, str]],
                     cc_addrs: List[Dict[str, str]], bcc_addrs: List[Dict[str, str]]) -> List[str]:
    emails: List[str] = []
    if from_addr.get("email"):
        emails.append(from_addr["email"].strip().lower())
    for arr in (to_addrs, cc_addrs, bcc_addrs):
        for a in arr:
            if a.get("email"):
                emails.append(a["email"].strip().lower())
    # unique + sorted
    uniq = sorted(set(e for e in emails if e))
    return uniq


def date_bucket_iso(date_utc_iso: Optional[str]) -> str:
    if not date_utc_iso:
        return "unknown"
    try:
        # Expect ISO8601; parse and bucket by day
        dt_obj = dt.datetime.fromisoformat(date_utc_iso.replace("Z", "+00:00"))
        dt_obj = dt_obj.astimezone(dt.timezone.utc)
        return dt_obj.date().isoformat()
    except Exception:
        return "unknown"


def build_fallback_conversation_id(subject_canon: str, participants: List[str], date_utc_iso: Optional[str]) -> str:
    bucket = date_bucket_iso(date_utc_iso)
    key = json.dumps({
        "subject": subject_canon.lower(),
        "participants": participants,
        "date_bucket": bucket
    }, sort_keys=True, ensure_ascii=False)
    return sha256_hex("fallback:" + key)


def canonical_message_id(mid: Optional[str]) -> Optional[str]:
    if not mid:
        return None
    mid = mid.strip().lower()
    mid = re.sub(r"^\s*<|>\s*$", "", mid)
    return mid or None


# -----------------------------
# IMAP Session with retry
# -----------------------------

@dataclass
class ImapConfig:
    host: str
    port: int
    user: str
    password: str
    ssl: bool
    mailbox: str
    readonly: bool = True


class ImapSession:
    def __init__(self, config: ImapConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.conn: Optional[imaplib.IMAP4] = None
        self.capabilities: Tuple[bytes, ...] = tuple()
        self.connect_and_login()
        self.select_mailbox()

    def connect_and_login(self) -> None:
        self.close()
        try:
            if self.config.ssl:
                self.conn = imaplib.IMAP4_SSL(self.config.host, self.config.port)
            else:
                self.conn = imaplib.IMAP4(self.config.host, self.config.port)
            typ, _ = self.conn.login(self.config.user, self.config.password)
            if typ != "OK":
                raise RuntimeError("Login failed")
            self.capabilities = getattr(self.conn, "capabilities", tuple())
        except Exception as e:
            self.close()
            raise e

    def select_mailbox(self) -> None:
        assert self.conn is not None
        typ, _ = self.conn.select(self.config.mailbox, readonly=self.config.readonly)
        if typ != "OK":
            raise RuntimeError(f"Failed to select mailbox {self.config.mailbox}")

    def close(self) -> None:
        if self.conn is not None:
            try:
                try:
                    self.conn.logout()
                except Exception:
                    try:
                        self.conn.shutdown()
                    except Exception:
                        pass
            finally:
                self.conn = None

    def _retry(self, func, *args, **kwargs):
        attempts = kwargs.pop("_attempts", 4)
        initial_delay = kwargs.pop("_initial_delay", 1.0)
        last_exc: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                return func(*args, **kwargs)
            except (imaplib.IMAP4.abort, imaplib.IMAP4.error, OSError, socket.error) as e:
                last_exc = e
                self.logger.warning(f"IMAP op failed (attempt {attempt}/{attempts}): {e.__class__.__name__}: {e}")
                # reconnect and reselect
                try:
                    time.sleep(initial_delay * (2 ** (attempt - 1)))
                    self.connect_and_login()
                    self.select_mailbox()
                except Exception as re_e:
                    last_exc = re_e
                    self.logger.warning(f"Reconnection failed: {re_e}")
                    time.sleep(initial_delay * (2 ** (attempt - 1)))
                continue
        if last_exc:
            raise last_exc
        raise RuntimeError("IMAP operation failed after retries")

    def uid(self, command: str, *args) -> Tuple[str, List[Any]]:
        assert self.conn is not None
        def do_uid():
            assert self.conn is not None
            return self.conn.uid(command, *args)
        typ, data = self._retry(do_uid, _attempts=4, _initial_delay=1.0)
        if typ != "OK":
            raise RuntimeError(f"UID {command} failed: {typ}")
        return typ, data


# -----------------------------
# Fetch and parse helpers
# -----------------------------

def chunked(iterable: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), size):
        yield list(iterable[i:i + size])


def build_uid_set(uids: List[str]) -> str:
    # Join as comma-separated; IMAP allows "1,2,3"
    return ",".join(uids)


UID_REGEX = re.compile(rb"UID\s+(\d+)")
FLAGS_REGEX = re.compile(rb"FLAGS\s+\(([^)]*)\)")
X_GM_THRID_REGEX = re.compile(rb"X-GM-THRID\s+(\d+)")


def parse_fetch_header_batch(data: List[Any]) -> List[Tuple[str, bytes]]:
    """
    Returns list of (uid, header_bytes).
    """
    results: List[Tuple[str, bytes]] = []
    for item in data:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], (bytes, bytearray)) and isinstance(item[1], (bytes, bytearray)):
            m = UID_REGEX.search(item[0])
            if not m:
                continue
            uid = m.group(1).decode("ascii", errors="replace")
            headers_bytes = bytes(item[1])
            results.append((uid, headers_bytes))
    return results


def parse_fetch_full_batch(data: List[Any]) -> List[Tuple[str, bytes, List[str], Optional[str]]]:
    """
    Returns list of (uid, raw_rfc822_bytes, flags_list, x_gm_thrid_or_none)
    """
    results: List[Tuple[str, bytes, List[str], Optional[str]]] = []
    for item in data:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], (bytes, bytearray)) and isinstance(item[1], (bytes, bytearray)):
            first = bytes(item[0])
            raw = bytes(item[1])
            m_uid = UID_REGEX.search(first)
            if not m_uid:
                continue
            uid = m_uid.group(1).decode("ascii", errors="replace")
            flags_list: List[str] = []
            m_flags = FLAGS_REGEX.search(first)
            if m_flags:
                flags_blob = m_flags.group(1).decode("utf-8", errors="replace").strip()
                if flags_blob:
                    for fl in flags_blob.split():
                        flags_list.append(fl)
            xgm_thrid: Optional[str] = None
            m_thr = X_GM_THRID_REGEX.search(first)
            if m_thr:
                xgm_thrid = m_thr.group(1).decode("ascii", errors="replace")
            results.append((uid, raw, flags_list, xgm_thrid))
    return results


def fetch_headers_for_uids(session: ImapSession, uids: List[str], header_fields: List[str]) -> List[Tuple[str, bytes]]:
    # IMAP requires space-separated header fields inside parens
    fields = " ".join(header_fields)
    query = f"(UID BODY.PEEK[HEADER.FIELDS ({fields})])"
    uid_set = build_uid_set(uids)
    _, data = session.uid("FETCH", uid_set, query)
    return parse_fetch_header_batch(data)


def fetch_full_for_uids(session: ImapSession, uids: List[str], include_gmail: bool) -> List[Tuple[str, bytes, List[str], Optional[str]]]:
    items = ["UID", "FLAGS", "RFC822"]
    if include_gmail:
        items.append("X-GM-THRID")
    query = "(" + " ".join(items) + ")"
    uid_set = build_uid_set(uids)
    _, data = session.uid("FETCH", uid_set, query)
    return parse_fetch_full_batch(data)


# -----------------------------
# Threading logic
# -----------------------------

@dataclass
class MinimalHeaderInfo:
    uid: str
    mailbox: str
    message_id: Optional[str]
    in_reply_to: Optional[str]
    references: List[str]
    subject: str
    subject_canonical: str
    from_addr: Dict[str, str]
    to_addrs: List[Dict[str, str]]
    cc_addrs: List[Dict[str, str]]
    bcc_addrs: List[Dict[str, str]]
    date_header: Optional[str]
    date_utc_iso: Optional[str]


def parse_headers_only(headers_bytes: bytes, uid: str, mailbox: str) -> MinimalHeaderInfo:
    msg = email.message_from_bytes(headers_bytes, policy=email.policy.default)
    # Extract headers
    subject = decode_mime_words(msg["Subject"])
    from_parsed = parse_addresses(msg.get("From"))
    from_addr = from_parsed[0] if from_parsed else {"name": "", "email": ""}
    to_addrs = parse_addresses(msg.get("To"))
    cc_addrs = parse_addresses(msg.get("Cc"))
    bcc_addrs = parse_addresses(msg.get("Bcc"))
    date_header, date_utc_iso = parse_date_to_utc(msg.get("Date"))
    message_id = canonical_message_id(extract_single_message_id(msg.get("Message-ID")))
    in_reply_to = canonical_message_id(extract_single_message_id(msg.get("In-Reply-To")))
    references = [canonical_message_id(r) or "" for r in extract_message_ids(msg.get("References"))]
    references = [r for r in references if r]
    subject_canonical = canonicalize_subject(subject)
    return MinimalHeaderInfo(
        uid=uid,
        mailbox=mailbox,
        message_id=message_id,
        in_reply_to=in_reply_to,
        references=references,
        subject=subject,
        subject_canonical=subject_canonical,
        from_addr=from_addr,
        to_addrs=to_addrs,
        cc_addrs=cc_addrs,
        bcc_addrs=bcc_addrs,
        date_header=date_header,
        date_utc_iso=date_utc_iso,
    )


def compute_root_message_id_for(uid: str,
                                info: MinimalHeaderInfo,
                                msgid_to_info: Dict[str, MinimalHeaderInfo]) -> Optional[str]:
    # If References present, root is the first reference
    if info.references:
        return info.references[0]
    # Else, walk In-Reply-To chain via known mapping
    visited: set = set()
    current = info.in_reply_to
    while current:
        if current in visited:
            break
        visited.add(current)
        parent = msgid_to_info.get(current)
        if not parent:
            # Parent not in this mailbox snapshot; treat current as root
            return current
        if parent.references:
            return parent.references[0]
        current = parent.in_reply_to
    # No headers to determine root
    return None


# -----------------------------
# Body and attachments parsing
# -----------------------------

@dataclass
class ParsedBody:
    text: str
    html: str


def decode_part_bytes(part: Message) -> bytes:
    try:
        payload = part.get_payload(decode=True)
        if payload is None:
            return b""
        return payload
    except Exception:
        # Sometimes incorrect encoding headers; attempt manual decoding
        payload = part.get_payload()
        if isinstance(payload, str):
            try:
                return payload.encode(part.get_content_charset() or "utf-8", errors="ignore")
            except Exception:
                return payload.encode("utf-8", errors="ignore")
        return b""


def decode_text_payload(part: Message) -> str:
    data = decode_part_bytes(part)
    charset = part.get_content_charset() or "utf-8"
    try:
        return data.decode(charset, errors="replace")
    except Exception:
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return data.decode("latin1", errors="replace")


def extract_bodies_and_attachments(msg: EmailMessage,
                                   attachments_dir: Optional[str],
                                   uid: str) -> Tuple[ParsedBody, List[Dict[str, Any]]]:
    body_text_candidates: List[str] = []
    body_html_candidates: List[str] = []
    attachments: List[Dict[str, Any]] = []

    part_index = 0
    for part in msg.walk():
        part_index += 1
        if part.is_multipart():
            continue
        content_type = (part.get_content_type() or "").lower()
        content_disposition = part.get_content_disposition()  # 'inline', 'attachment', or None
        filename = part.get_filename()
        filename = sanitize_filename(filename) if filename else None
        content_id = part.get("Content-ID")
        cid_norm = None
        if content_id:
            m = re.match(r"^\s*<(.+?)>\s*$", content_id.strip())
            cid_norm = (m.group(1) if m else content_id).strip()

        is_attachment = False
        if content_disposition in ("attachment", "inline"):
            is_attachment = True
        # If filename present, treat as attachment even if disposition is None
        if filename and content_disposition is None:
            is_attachment = True

        # Collect bodies
        if not is_attachment:
            if content_type == "text/plain":
                text = decode_text_payload(part)
                if text.strip():
                    body_text_candidates.append(text)
            elif content_type == "text/html":
                html = decode_text_payload(part)
                if html.strip():
                    body_html_candidates.append(html)

        # Save attachments (including inline)
        if is_attachment:
            content_bytes = decode_part_bytes(part)
            content_sha256 = hashlib.sha256(content_bytes).hexdigest()
            size = len(content_bytes)
            saved_path = None
            if attachments_dir:
                try:
                    os.makedirs(attachments_dir, exist_ok=True)
                    safe_name = filename or f"attachment_{uid}_{part_index}"
                    base, ext = os.path.splitext(safe_name)
                    # Ensure uniqueness
                    target = os.path.join(attachments_dir, f"{base}_{uid}_{part_index}{ext}")
                    with open(target, "wb") as f:
                        f.write(content_bytes)
                    saved_path = target
                except Exception:
                    saved_path = None
            attachments.append({
                "filename": filename or "",
                "content_type": content_type or "",
                "size": size,
                "sha256": content_sha256,
                "saved_path": saved_path,
                "content_id": cid_norm
            })

    body_text = ""
    body_html = ""
    if body_text_candidates:
        # Prefer the first text/plain
        body_text = body_text_candidates[0]
    elif body_html_candidates:
        body_html = body_html_candidates[0]
        # Generate text from HTML if only HTML present
        body_text = html_to_text(body_html)

    return ParsedBody(text=body_text or "", html=body_html or ""), attachments


# -----------------------------
# Main export logic
# -----------------------------

def export_mailbox(config: ImapConfig,
                   out_path: str,
                   limit: Optional[int],
                   attachments_dir: Optional[str],
                   batch_size: int,
                   logger: logging.Logger) -> None:
    session = ImapSession(config, logger)
    include_gmail = any(c == b"X-GM-EXT-1" for c in session.capabilities)

    # 1) Get all UIDs
    _, data = session.uid("SEARCH", None, "ALL")
    if not data or not data[0]:
        logger.info("No messages found.")
        return
    raw_uids = data[0].decode("ascii", errors="replace").strip().split()
    # UID list as strings
    all_uids = raw_uids
    if limit is not None and limit > 0:
        all_uids = all_uids[:limit]

    total = len(all_uids)
    logger.info(f"Enumerating {total} messages from mailbox '{config.mailbox}'")

    # 2) Pass 1: fetch minimal headers to compute threading and metadata
    header_fields = ["Message-ID", "In-Reply-To", "References", "Subject", "From", "To", "Cc", "Bcc", "Date"]
    uid_to_info: Dict[str, MinimalHeaderInfo] = {}
    msgid_to_info: Dict[str, MinimalHeaderInfo] = {}

    processed = 0
    for batch_uids in chunked(all_uids, batch_size):
        # Fetch headers for batch
        try:
            batch_headers = fetch_headers_for_uids(session, batch_uids, header_fields)
        except Exception as e:
            logger.warning(f"Header fetch batch failed; retrying smaller or skipping. Error: {e}")
            # retry one by one
            for u in batch_uids:
                try:
                    single_headers = fetch_headers_for_uids(session, [u], header_fields)
                except Exception as e2:
                    logger.error(f"Failed to fetch headers for UID {u}: {e2}")
                    continue
                for (uid, headers_bytes) in single_headers:
                    info = parse_headers_only(headers_bytes, uid, config.mailbox)
                    uid_to_info[uid] = info
                    if info.message_id:
                        msgid_to_info[info.message_id] = info
            processed += len(batch_uids)
            logger.info(f"Header pass progress: {processed}/{total}")
            continue

        for (uid, headers_bytes) in batch_headers:
            info = parse_headers_only(headers_bytes, uid, config.mailbox)
            uid_to_info[uid] = info
            if info.message_id:
                msgid_to_info[info.message_id] = info

        processed += len(batch_uids)
        logger.info(f"Header pass progress: {processed}/{total}")

    # 3) Compute conversation roots and IDs
    uid_to_conversation_id: Dict[str, str] = {}
    for uid, info in uid_to_info.items():
        root_mid = compute_root_message_id_for(uid, info, msgid_to_info)
        if root_mid:
            conv_id = sha256_hex("root:" + root_mid)
        else:
            parts = participants_set(info.from_addr, info.to_addrs, info.cc_addrs, info.bcc_addrs)
            conv_id = build_fallback_conversation_id(info.subject_canonical, parts, info.date_utc_iso)
        uid_to_conversation_id[uid] = conv_id

    # 4) Pass 2: fetch full messages and write NDJSON
    fetch_ts_utc = dt.datetime.now(tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    count_written = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        processed = 0
        for batch_uids in chunked(all_uids, batch_size):
            try:
                batch_full = fetch_full_for_uids(session, batch_uids, include_gmail=include_gmail)
            except Exception as e:
                logger.warning(f"Full fetch batch failed; retrying singly. Error: {e}")
                batch_full = []
                for u in batch_uids:
                    try:
                        single_full = fetch_full_for_uids(session, [u], include_gmail=include_gmail)
                        batch_full.extend(single_full)
                    except Exception as e2:
                        logger.error(f"Failed to fetch full message for UID {u}: {e2}")
                        continue

            for (uid, raw_bytes, flags, xgm_thrid) in batch_full:
                try:
                    msg = email.message_from_bytes(raw_bytes, policy=email.policy.default)
                except Exception as e:
                    logger.error(f"Failed to parse message UID {uid}: {e}")
                    continue

                # Prefer header info from pass 1 to ensure consistency
                info = uid_to_info.get(uid)
                if info is None:
                    # Fallback: parse from full message
                    subject = decode_mime_words(msg.get("Subject"))
                    from_parsed = parse_addresses(msg.get("From"))
                    from_addr = from_parsed[0] if from_parsed else {"name": "", "email": ""}
                    to_addrs = parse_addresses(msg.get("To"))
                    cc_addrs = parse_addresses(msg.get("Cc"))
                    bcc_addrs = parse_addresses(msg.get("Bcc"))
                    date_header, date_utc_iso = parse_date_to_utc(msg.get("Date"))
                    message_id = canonical_message_id(extract_single_message_id(msg.get("Message-ID")))
                    in_reply_to = canonical_message_id(extract_single_message_id(msg.get("In-Reply-To")))
                    references = [canonical_message_id(r) or "" for r in extract_message_ids(msg.get("References"))]
                    references = [r for r in references if r]
                    subject_canonical = canonicalize_subject(subject)
                    info = MinimalHeaderInfo(
                        uid=uid,
                        mailbox=config.mailbox,
                        message_id=message_id,
                        in_reply_to=in_reply_to,
                        references=references,
                        subject=subject,
                        subject_canonical=subject_canonical,
                        from_addr=from_addr,
                        to_addrs=to_addrs,
                        cc_addrs=cc_addrs,
                        bcc_addrs=bcc_addrs,
                        date_header=date_header,
                        date_utc_iso=date_utc_iso,
                    )

                conv_id = uid_to_conversation_id.get(uid)
                if not conv_id:
                    parts = participants_set(info.from_addr, info.to_addrs, info.cc_addrs, info.bcc_addrs)
                    conv_id = build_fallback_conversation_id(info.subject_canonical, parts, info.date_utc_iso)

                bodies, attachments = extract_bodies_and_attachments(msg, attachments_dir, uid)

                record: Dict[str, Any] = {
                    "uid": uid,
                    "mailbox": info.mailbox,
                    "message_id": info.message_id,
                    "conversation_id": conv_id,
                    "subject": info.subject,
                    "subject_canonical": info.subject_canonical,
                    "date_header": info.date_header,
                    "date_utc": info.date_utc_iso,
                    "from": info.from_addr,
                    "to": info.to_addrs,
                    "cc": info.cc_addrs,
                    "bcc": info.bcc_addrs,
                    "in_reply_to": info.in_reply_to,
                    "references": info.references,
                    "body_text": bodies.text,
                    "body_html": bodies.html,
                    "attachments": attachments,
                    "flags": flags,
                    "fetch_ts_utc": fetch_ts_utc,
                    "provider_thread_id": xgm_thrid if xgm_thrid else None
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count_written += 1

            processed += len(batch_uids)
            logger.info(f"Export progress: {processed}/{total} (written: {count_written})")

    logger.info(f"Done. Wrote {count_written} messages to {out_path}")


# -----------------------------
# Argparse and main
# -----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export IMAP mailbox to NDJSON with stable conversation IDs."
    )
    parser.add_argument("--out", required=True, help="Output NDJSON file path, e.g., messages.ndjson")
    parser.add_argument("--mailbox", default=os.environ.get("IMAP_MAILBOX", "INBOX"),
                        help="Mailbox name to select (default: INBOX or IMAP_MAILBOX env)")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of messages for testing")
    parser.add_argument("--attachments-dir", default=None, help="Directory to save attachments (optional)")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for UID FETCH (default: 500)")
    parser.add_argument("--ssl", type=str, default=None,
                        help="Use SSL (true/false). Overrides SSL env var if provided. Default: true")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser.parse_args(argv)


def setup_logger(level_str: str) -> logging.Logger:
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    return logging.getLogger("imap_export")


def read_imap_config(args: argparse.Namespace, logger: logging.Logger) -> ImapConfig:
    host = os.environ.get("IMAP_HOST")
    port_str = os.environ.get("IMAP_PORT")
    user = os.environ.get("IMAP_USER")
    password = os.environ.get("IMAP_PASSWORD")

    if not host or not port_str or not user or not password:
        logger.error("Missing IMAP environment variables. Required: IMAP_HOST, IMAP_PORT, IMAP_USER, IMAP_PASSWORD")
        sys.exit(1)

    try:
        port = int(port_str)
    except Exception:
        logger.error("Invalid IMAP_PORT (must be integer)")
        sys.exit(1)

    ssl_from_env = env_bool("SSL", True)
    ssl_flag = to_bool(args.ssl, ssl_from_env) if args.ssl is not None else ssl_from_env

    mailbox = args.mailbox or os.environ.get("IMAP_MAILBOX", "INBOX")
    if not mailbox:
        mailbox = "INBOX"

    return ImapConfig(
        host=host,
        port=port,
        user=user,
        password=password,
        ssl=ssl_flag,
        mailbox=mailbox,
        readonly=True
    )


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger(args.log_level)

    try:
        config = read_imap_config(args, logger)
        export_mailbox(
            config=config,
            out_path=args.out,
            limit=args.limit,
            attachments_dir=args.attachments_dir,
            batch_size=max(1, args.batch_size),
            logger=logger
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


