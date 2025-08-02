import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

# 1. Auth
scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)

# 2. Open sheet
sheet = client.open("Kenya Cost of Living").sheet1  # or use .worksheet("Data")

# 3. Load CSV
df = pd.read_csv("data/cost_Data.csv")

# 4. Clear and update
sheet.clear()
sheet.update([df.columns.values.tolist()] + df.values.tolist())
