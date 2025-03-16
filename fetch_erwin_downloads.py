import requests
import json
from datetime import datetime

def get_total_downloads(package_name):
    try:
        # Get package metadata to find creation date
        registry_url = f"https://registry.npmjs.org/{package_name}"
        response = requests.get(registry_url)
        response.raise_for_status()
        
        package_data = response.json()
        creation_date = package_data['time']['created'][:7]  # Get YYYY-MM format
        
        # Get download stats from creation to current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        stats_url = f"https://api.npmjs.org/downloads/range/{creation_date}-01:{current_date}/{package_name}"
        
        stats_response = requests.get(stats_url)
        stats_response.raise_for_status()
        
        stats_data = stats_response.json()
        return sum(day['downloads'] for day in stats_data['downloads'])
    
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
    except KeyError as e:
        print(f"Missing expected data in API response: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None

def save_downloads(data):
    try:
        with open('downloads_erwin_badge.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False, sort_keys=True)
        print("Download data saved successfully")
    except PermissionError:
        print("Error: No write permissions for the file")
    except Exception as e:
        print(f"Failed to save data: {str(e)}")

# Main execution
if __name__ == "__main__":
    package_name = "@__haoyang__/erwin"
    total_downloads = get_total_downloads(package_name)
    
    if total_downloads is not None:
        badge_data = {
            "schemaVersion": 1,
            "label": "Downloads",
            "message": f"{total_downloads:,}",
            "color": "blue",
            "lastUpdated": datetime.now().isoformat()
        }
        save_downloads(badge_data)
        print(f"Total downloads for {package_name}: {total_downloads:,}")
    else:
        print(f"Failed to fetch download count for {package_name}")
