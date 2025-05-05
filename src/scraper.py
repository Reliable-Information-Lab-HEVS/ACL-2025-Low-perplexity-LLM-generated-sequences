import base64
import urllib.parse
import requests
import json
import sys

def post_to_endpoint(message, base_url="https://your-website.com/"):
    """
    Makes a direct POST request to the server's query endpoint
    
    Args:
        message (str): The message to process
        base_url (str): The base URL of your website
    """
    # Prepare the endpoint URL
    endpoint_url = f"{base_url.rstrip('/')}/q"
    
    # Create the payload as expected by the server
    payload = {
        'document': message,
        'seq_id': 1  # The code increments this for each request
    }
    
    print(f"POSTing to endpoint: {endpoint_url}")
    print(f"Payload length: {len(message)} characters")
    
    # Make the POST request
    response = requests.post(
        endpoint_url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json"
        },
        data=json.dumps(payload)
    )
    
    # Check if successful
    if response.status_code == 200:
        try:
            result = response.json()
            print("Server response received:")
            
            # Print key results
            if 'chains' in result and len(result['chains']) > 0:
                print("\nLongest matches:")
                for i, chain in enumerate(result['chains'][:5]):  # Show top 5
                    print(f"{i+1}. {''.join(chain)}")
            else:
                print("No matches found in the response")
            
            # Save full response to file
            with open('full_response.json', 'w') as f:
                json.dump(result, f, indent=2)
            print("\nFull response saved to full_response.json")
            
            return result
        except json.JSONDecodeError:
            print("Error: Server response is not valid JSON")
            print(response.text[:1000])  # Print first 1000 chars
    else:
        print(f"Error: Server returned status code {response.status_code}")
        print(response.text[:1000])  # Print first 1000 chars
    
    return None

def also_update_url_hash(message, base_url="https://your-website.com/"):
    """
    Also updates the URL with hash (mainly for documentation, not required for function)
    
    Args:
        message (str): The message to encode
        base_url (str): The base URL of your website
    """
    # URL encode and then base64 encode
    url_encoded = urllib.parse.quote(message)
    base64_encoded = base64.b64encode(url_encoded.encode()).decode()
    
    # Create the full URL with hash
    full_url = f"{base_url}#{base64_encoded}"
    print(f"Corresponding URL with hash would be: {full_url}")
    
    return full_url

if __name__ == "__main__":
    # Use command line argument if provided, otherwise use a default message
    message = sys.argv[1] if len(sys.argv) > 1 else "Bonjour, c'est un test."
    
    # Replace with your actual website URL
    website_url = "https://pile.dataportraits.org/"
    
    # Generate and show the URL with hash (for reference)
    also_update_url_hash(message, website_url)
    
    # Make the direct POST request to the endpoint
    post_to_endpoint(message, website_url)