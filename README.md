# Smart Mail Extractor

A powerful Python tool for extracting emails from IMAP servers with intelligent conversation threading, attachment handling, and structured data export.

## Features

- **IMAP Integration**: Connect to any IMAP-compatible email server
- **Conversation Threading**: Automatically groups related emails using Message-ID references
- **Multiple Mailbox Support**: Process multiple mailboxes in a single run
- **Attachment Extraction**: Save email attachments to local directories
- **Structured Export**: Output emails in NDJSON format for easy processing
- **HTML to Text Conversion**: Extract clean text content from HTML emails
- **Batch Processing**: Efficient handling of large email volumes
- **Configurable**: Environment variables and command-line options for flexibility

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <https://github.com/grechkinet/smart_mail_extractor>
cd smart_mail_extractor
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Set the following environment variables for your IMAP server:

```bash
export IMAP_HOST="imap.gmail.com"
export IMAP_PORT="993"
export IMAP_USER="your-email@gmail.com"
export IMAP_PASSWORD="your-app-password"
export IMAP_MAILBOX="INBOX"
export SSL="true"
```

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `IMAP_HOST` | IMAP server hostname | - | Yes |
| `IMAP_PORT` | IMAP server port | - | Yes |
| `IMAP_USER` | Email username | - | Yes |
| `IMAP_PASSWORD` | Email password/app password | - | Yes |
| `IMAP_MAILBOX` | Default mailbox name | `INBOX` | No |
| `SSL` | Use SSL connection | `true` | No |

## Usage

### Basic Export

Export emails from the default mailbox (INBOX):

```bash
python export_emails.py --out messages.ndjson
```

### Export from Specific Mailbox

```bash
python export_emails.py --out messages.ndjson --mailbox "Sent"
```

### Export from Multiple Mailboxes

```bash
python export_emails.py --out messages.ndjson --mailboxes "INBOX,Sent,Archive"
```

### Include Attachments

```bash
python export_emails.py --out messages.ndjson --attachments-dir ./attachments
```

### Limit Number of Messages (for testing)

```bash
python export_emails.py --out messages.ndjson --limit 100
```

### Advanced Options

```bash
python export_emails.py \
  --out messages.ndjson \
  --mailboxes "INBOX,Sent" \
  --attachments-dir ./attachments \
  --batch-size 1000 \
  --log-level DEBUG
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--out` | Output NDJSON file path | Required |
| `--mailbox` | Single mailbox to process | `INBOX` or `IMAP_MAILBOX` env |
| `--mailboxes` | Comma-separated list of mailboxes | Overrides `--mailbox` |
| `--limit` | Maximum number of messages to export | No limit |
| `--attachments-dir` | Directory to save attachments | No attachments |
| `--batch-size` | IMAP fetch batch size | 500 |
| `--ssl` | Use SSL connection | `true` or `SSL` env |
| `--log-level` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |

## Output Format

The tool exports emails in NDJSON (Newline Delimited JSON) format. Each line contains a JSON object representing an email with the following structure:

```json
{
  "uid": "12345",
  "mailbox": "INBOX",
  "message_id": "abc123@example.com",
  "conversation_id": "conv_abc123",
  "subject": "Email Subject",
  "from": {"name": "Sender Name", "email": "sender@example.com"},
  "to": [{"name": "Recipient", "email": "recipient@example.com"}],
  "cc": [],
  "bcc": [],
  "date": "2024-01-15T10:30:00Z",
  "text_content": "Plain text content...",
  "html_content": "<html>...</html>",
  "attachments": [
    {
      "filename": "document.pdf",
      "content_type": "application/pdf",
      "size": 1024
    }
  ]
}
```

## Conversation Threading

The tool automatically groups related emails using:
1. **Message-ID references**: Links emails in reply chains
2. **Subject canonicalization**: Groups emails with similar subjects
3. **Participant matching**: Identifies conversations by participants
4. **Date bucketing**: Organizes emails by time periods

## Examples

### Gmail Export

```bash
export IMAP_HOST="imap.gmail.com"
export IMAP_PORT="993"
export IMAP_USER="your-email@gmail.com"
export IMAP_PASSWORD="your-app-password"
export SSL="true"

python export_emails.py --out gmail_export.ndjson --mailboxes "INBOX,Sent"
```

### Outlook/Office 365 Export

```bash
export IMAP_HOST="outlook.office365.com"
export IMAP_PORT="993"
export IMAP_USER="your-email@outlook.com"
export IMAP_PASSWORD="your-password"
export SSL="true"

python export_emails.py --out outlook_export.ndjson --mailbox "INBOX"
```

### Yahoo Mail Export

```bash
export IMAP_HOST="imap.mail.yahoo.com"
export IMAP_PORT="993"
export IMAP_USER="your-email@yahoo.com"
export IMAP_PASSWORD="your-app-password"
export SSL="true"

python export_emails.py --out yahoo_export.ndjson --mailbox "INBOX"
```

## Troubleshooting

### Common Issues

1. **Authentication Failed**: Ensure you're using the correct password or app password
2. **SSL Connection Error**: Check if your server requires SSL and verify the port
3. **Permission Denied**: Some servers require enabling IMAP access in settings
4. **Rate Limiting**: Use `--batch-size` to reduce server load

### Debug Mode

Enable detailed logging to troubleshoot issues:

```bash
python export_emails.py --out debug.ndjson --log-level DEBUG
```

## Security Notes

- Store credentials securely using environment variables
- Use app passwords instead of regular passwords when possible
- The tool operates in read-only mode by default
- Avoid committing credentials to version control

## Dependencies

- `beautifulsoup4`: HTML parsing and text extraction
- `lxml`: XML/HTML processing backend
- Standard Python libraries: `imaplib`, `email`, `json`, etc.

## License

This project is licensed under the terms specified in the LICENSE file.
