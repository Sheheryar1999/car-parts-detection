import requests

ngrok_url = " https://8503-73-171-225-84.ngrok-free.app"
file_path = r"C:\Users\shehe\Desktop\Project\CarPartsDetectionChallenge-master-working\00229.jpg"
files = {"file": open(file_path, "rb")}

response = requests.post(ngrok_url, files=files)
print(response.json())
