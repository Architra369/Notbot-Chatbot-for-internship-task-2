from twilio.rest import Client
account_sid = 'ACf445c26885609b3cf7cf03126a461ff9'
auth_token = 'dfad6b447e7597d4b139ebb76100f8f4'
client = Client(account_sid, auth_token)
def send_rem(date,rem):
  message=client.messages.create(
  from_='whatsapp:+14155238886',
  body='*REMINDER* '+date+'\n'+rem,
  to='whatsapp:+919793619302')
  print(message.sid)