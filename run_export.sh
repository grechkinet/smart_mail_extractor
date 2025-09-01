#!/bin/bash

# Переходим в директорию скрипта
cd /Users/maksim/Documents/WU_WINF/Bachelor/smart_email_export/smart_mail_extractor

# Активируем виртуальное окружение
source .venv/bin/activate

# Устанавливаем переменные окружения
export IMAP_HOST="mail.your-server.de"
export IMAP_PORT="993"
export IMAP_USER="test@reisigo.com"
export IMAP_PASSWORD="aX97QKtm-tqLuxs"
export SSL="true"
export AGENCY_DOMAINS="reisigo.com"
export TENANT="reisigo_prod"

# Запускаем экспорт
python3 export_emails.py \
  --out messages_with_session.ndjson \
  --mailboxes "INBOX,INBOX.Sent" \
  --batch-size 200 \
  --log-level DEBUG
