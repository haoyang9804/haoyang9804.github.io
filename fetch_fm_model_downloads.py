import requests
import json

organization = "fm-universe"
base_url = f"https://huggingface.co/api/models?author={organization}&per_page=100"
total_downloads = 0

try:
    next_url = base_url
    while next_url:
        response = requests.get(next_url)
        response.raise_for_status()
        
        # Process current page
        models = response.json()
        for model in models:
            total_downloads += model.get("downloads", 0)
        
        # Check for next page in Link header
        link_header = response.headers.get('Link', '')
        next_link = [link for link in link_header.split(', ') 
                    if 'rel="next"' in link]
        
        next_url = next_link[0].split(';')[0][1:-1] if next_link else None

except requests.exceptions.RequestException as e:
    print(f"Error fetching data: {e}")
    exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    exit(1)

# Generate badge (same as before)
badge_data = {
    "schemaVersion": 1,
    "label": "Downloads",
    "message": f"{total_downloads:,}",
    "color": "blue"
}

with open("downloads_fm_badge.json", "w") as f:
    json.dump(badge_data, f)
