import requests

def recommend_papers(query_text: str):
    """
    Recommends similar research papers using the Semantic Scholar API.
    """
    try:
        # Using Semantic Scholar's Keyword Search API
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query_text)}&limit=5&fields=title,authors,url"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        results = response.json()

        if "data" in results and results["data"]:
            papers = []
            for item in results["data"]:
                papers.append({
                    "title": item.get("title", "No Title"),
                    "authors": [author["name"] for author in item.get("authors", [])],
                    "url": item.get("url", "#")
                })
            return papers
        else:
            return []

    except requests.exceptions.RequestException as e:
        print(f"Error fetching recommendations: {e}")
        return []
