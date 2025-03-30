import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email Credentials
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587  # Use 587 for TLS, 465 for SSL
EMAIL_SENDER = "shrivastavavedansh082@gmail.com"
EMAIL_PASSWORD = "dxrwpelprztrmqdv"
EMAIL_RECEIVER = "sonalisengar580@gmail.com"

def get_location():
    """Fetches the approximate location using an IP-based API."""
    try:
        response = requests.get("https://ipinfo.io/json").json()
        location = f"{response.get('city', 'Unknown')}, {response.get('region', 'Unknown')}, {response.get('country', 'Unknown')}"
        return f"📍 Current Location: {location} (Approximate)"
    except Exception as e:
        print(f"⚠️ Location retrieval failed: {e}")
        return "⚠️ Location unavailable"

def send_email(location):
    """Sends an emergency email with location details."""
    subject = "🚨 Emergency Alert: Blind Person Needs Help!"
    body = f"An emergency has been detected. A blind person needs help.\n\n{location}"

    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Secure connection
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())

        print("📧 Emergency email sent successfully!")
    except Exception as e:
        print(f"❌ Email sending failed: {e}")