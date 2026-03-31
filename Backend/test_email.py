"""
Run this to test if your Gmail App Password is working.
Usage: python test_email.py
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── FILL THESE IN ──────────────────────────────────────
SENDER_EMAIL    = 'immoksha7@gmail.com'       # your Gmail
SENDER_PASSWORD = 'lrjq kueh wibf wwms'       # your 16-char App Password
RECIPIENT_EMAILS = [
    'immoksha7@gmail.com',
    'yubikachaudhary@gmail.com',
]
# ───────────────────────────────────────────────────────

print("Testing Gmail connection...")
print(f"Sending FROM : {SENDER_EMAIL}")
print(f"Sending TO   : {RECIPIENT_EMAILS}")
print()

try:
    msg = MIMEMultipart()
    msg['From']    = SENDER_EMAIL
    msg['To']      = ', '.join(RECIPIENT_EMAILS)
    msg['Subject'] = '✅ DermaScan Email Test'
    msg.attach(MIMEText(
        "This is a test email from DermaScan.\n\n"
        "If you received this, your email setup is working correctly! 🎉",
        'plain'
    ))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        print("Connecting to Gmail SMTP...")
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        print("✅ Login successful!")
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAILS, msg.as_string())
        print("✅ Email sent successfully!")
        print()
        print("Check both inboxes:")
        for r in RECIPIENT_EMAILS:
            print(f"  → {r}")

except smtplib.SMTPAuthenticationError:
    print("❌ Authentication failed!")
    print()
    print("This means your App Password is wrong. Here's how to fix it:")
    print("1. Go to https://myaccount.google.com")
    print("2. Security → 2-Step Verification → make sure it's ON")
    print("3. Security → App Passwords")
    print("4. Select app: Mail, device: Windows Computer")
    print("5. Click Generate → copy the 16-character password")
    print("6. Paste it in this file AND in app.py as SENDER_PASSWORD")
    print()
    print("⚠️  Common mistakes:")
    print("   - Using your real Gmail password instead of the App Password")
    print("   - App Password has spaces (e.g. 'abcd efgh ijkl mnop') — include them as-is")
    print("   - 2-Step Verification is not turned on")

except smtplib.SMTPException as e:
    print(f"❌ SMTP Error: {e}")

except Exception as e:
    print(f"❌ Error: {e}")