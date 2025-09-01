#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import imaplib
import os
import sys
import re
from typing import List, Tuple

# -----------------------------
# Utils
# -----------------------------

def env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

def mask_login(login: str) -> str:
    if "@" in login:
        name, dom = login.split("@", 1)
        if len(name) <= 1:
            return "*" * len(name) + "@" + dom
        return name[0] + "***@" + dom
    return login[:1] + "***"

def parse_mailbox_name(raw_line: bytes) -> str:
    """
    Парсит имя ящика из строки ответа LIST.
    Обычно формат такой:
      b'(\\HasNoChildren) "." "INBOX/Sub"'
    Последний кавычённый элемент — имя.
    """
    s = raw_line.decode("utf-8", errors="replace")
    quoted = re.findall(r'"((?:[^"\\]|\\.)*)"', s)
    name = quoted[-1] if quoted else s.split()[-1].strip('"')
    # Декод из IMAP UTF-7 (для юникода в именах)
    try:
        name = imaplib.IMAP4._decode_utf7(name)
    except Exception:
        pass
    return name

def list_mailboxes(mail: imaplib.IMAP4) -> List[Tuple[str, bytes]]:
    typ, mailboxes = mail.list()
    if typ != "OK" or mailboxes is None:
        raise RuntimeError("Не удалось получить список mailboxes (LIST).")
    return [(parse_mailbox_name(m), m) for m in mailboxes]

def count_messages(mail: imaplib.IMAP4, mailbox_name: str) -> int:
    typ_sel, _ = mail.select(f'"{mailbox_name}"', readonly=True)
    if typ_sel != "OK":
        return -1
    typ_s, data = mail.search(None, "ALL")
    if typ_s != "OK" or not data or data[0] is None:
        return 0
    ids_blob = data[0]
    return 0 if not ids_blob else len(ids_blob.split())

def connect_imap(host: str, port: int, use_ssl: bool) -> imaplib.IMAP4:
    return imaplib.IMAP4_SSL(host, port) if use_ssl else imaplib.IMAP4(host, port)

def ensure_auth(mail: imaplib.IMAP4, user: str, password: str) -> None:
    """
    Проводит обычный LOGIN. Если сервер отвечает "Re-Authentication Failure",
    пересоздаём соединение — и пробуем ещё раз (один ретрай).
    """
    # Если каким-то чудом уже AUTH — просто выходим
    if getattr(mail, "state", None) in ("AUTH", "SELECTED"):
        return
    try:
        mail.login(user, password)
    except imaplib.IMAP4.error as e:
        msg = str(e)
        if "Re-Authentication Failure" in msg:
            # Это не успех, а ошибка. Пересоздаём соединение и пробуем ещё раз.
            raise  # поднимем выше — снаружи сделаем ретрай на новом соединении
        raise

# -----------------------------
# Main
# -----------------------------

def check_mailboxes():
    host = os.environ.get("IMAP_HOST", "mail.your-server.de")
    port = int(os.environ.get("IMAP_PORT", "993"))
    user = os.environ.get("IMAP_USER", "test@reisigo.com")
    password = os.environ.get("IMAP_PASSWORD", "aX97QKtm-tqLuxs")
    use_ssl = env_bool("SSL", True)

    print(f"Подключение к {host}:{port} как {mask_login(user)}")
    print(f"SSL: {use_ssl}")
    print("-" * 50)

    mail = None
    try:
        # 1) Подключаемся
        mail = connect_imap(host, port, use_ssl)
        welcome = getattr(mail, "welcome", b"")
        wtxt = welcome.decode("utf-8", errors="ignore") if isinstance(welcome, (bytes, bytearray)) else str(welcome)
        print(f"Приветствие сервера: {wtxt.strip()}")
        print(f"Начальное состояние: {getattr(mail, 'state', None)}")

        # 2) Логинимся (первая попытка)
        try:
            ensure_auth(mail, user, password)
        except imaplib.IMAP4.error as e:
            if "Re-Authentication Failure" in str(e):
                # пересоздаём коннект и пробуем ещё раз
                try:
                    mail.shutdown()  # безопасно закрыть сокет, если есть
                except Exception:
                    pass
                mail = connect_imap(host, port, use_ssl)
                ensure_auth(mail, user, password)  # если опять упадёт — выйдем ниже
            else:
                raise

        # 3) Проверяем состояние
        state = getattr(mail, "state", None)
        if state not in ("AUTH", "SELECTED"):
            # Последняя страховка: если сервер не перевёл в AUTH — это провал аутентификации.
            raise imaplib.IMAP4.error(f"Неавторизованное состояние после LOGIN: {state}")

        print("✅ Сеанс авторизован.\n")

        # (опционально) CAPABILITY
        try:
            typ_cap, caps = mail.capability()
            if typ_cap == "OK":
                caps_joined = b" ".join(caps) if isinstance(caps, list) else caps
                print(f"Возможности сервера: {caps_joined.decode('utf-8', errors='ignore')}\n")
        except Exception:
            pass

        # 4) LIST -> SELECT -> SEARCH
        print("📁 Доступные mailboxes:")
        print("-" * 30)

        items = list_mailboxes(mail)
        mailbox_names: List[str] = []

        for name, raw in items:
            mailbox_names.append(name)
            print(f"  📂 {name}")

            cnt = count_messages(mail, name)
            if cnt < 0:
                raw_str = raw.decode("utf-8", errors="replace")
                print(f"     └─ Ошибка SELECT (имя: {repr(name)}). LIST: {raw_str}")
            else:
                print(f"     └─ Сообщений: {cnt}")
            print()

        print("-" * 30)
        print(f"Всего mailboxes: {len(mailbox_names)}\n")
        if mailbox_names:
            preview = ",".join(mailbox_names[:5])
            print("💡 Для использования в скрипте (пример):")
            print(f'export IMAP_MAILBOXES="{preview}"')

        mail.logout()

    except imaplib.IMAP4.abort as e:
        print(f"❌ Соединение прервано сервером: {e}")
        sys.exit(1)
    except imaplib.IMAP4.error as e:
        print("❌ Ошибка IMAP:", e)
        # Диагностика причин
        hint = []
        hint.append("• Проверь логин/пароль (точные, без лишних пробелов).")
        hint.append("• На некоторых серверах нужен отдельный 'App password' для IMAP.")
        hint.append("• Возможен лимит попыток/блокировка после неудачных логинов — подождать/сбросить пароль.")
        hint.append("• Убедись, что IMAP включён в панели провайдера.")
        print("Возможные причины:\n" + "\n".join(hint))
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        sys.exit(1)
    finally:
        try:
            if mail is not None:
                mail.logout()
        except Exception:
            pass

if __name__ == "__main__":
    # Переменные окружения:
    #   export IMAP_HOST=mail.your-server.de
    #   export IMAP_PORT=993
    #   export IMAP_USER="test@reisigo.com"
    #   export IMAP_PASSWORD="********"
    #   export SSL=true
    check_mailboxes()