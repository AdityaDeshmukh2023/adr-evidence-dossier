import os  # For environment variable storage
import concurrent.futures  # Functions for parallel execution of tasks
import requests  # For making HTTP requests
import tiktoken  # For tokenization (optional, depending on token limits)

def fetch_food_context(keys, max_tokens=128000, max_workers=5, timeout=5):
    """
    Fetches ingredient data from FoodData Central API using multithreading and ensures the
    combined context does not exceed the specified token limit.

    Args:
        keys (list): List of ingredient names to search for.
        max_tokens (int): Maximum number of tokens allowed in the combined context.
        max_workers (int): Number of threads to use for concurrent processing.
        timeout (int): Timeout for each HTTP request in seconds.

    Returns:
        str: Combined context with summaries of the ingredients.
    """

    # Load the API key from environment variables
    api_key = os.getenv("FOODDATA_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Set the FOODDATA_API_KEY environment variable.")

    def get_food_data(key):
        url = f"https://api.nal.usda.gov/fdc/v1/foods/search"
        params = {
            "query": key,
            "pageSize": 1,  # Fetch only the top result
            "api_key": api_key,
        }
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for key '{key}': {e}")
            return None

    # Initialize the tokenizer for token counting (if needed)
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Container for ingredient summaries
    summaries = []

    # Multithreading to process keys
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {executor.submit(get_food_data, key): key for key in keys}
        for future in concurrent.futures.as_completed(future_to_key):
            try:
                data = future.result()
                if data and "foods" in data:
                    for food in data["foods"]:
                        description = food.get("description", "")
                        brand = food.get("brandOwner", "Unknown brand")
                        nutrients = food.get("foodNutrients", [])

                        nutrient_summary = ", ".join(
                            f"{nutrient['nutrientName']}: {nutrient['value']} {nutrient['unitName']}"
                            for nutrient in nutrients[:3]  # Limit to 3 key nutrients for brevity
                        )

                        summaries.append(
                            f"{description} (Brand: {brand}). Key nutrients: {nutrient_summary}."
                        )
            except Exception as e:
                key = future_to_key[future]
                print(f"Error processing key '{key}': {e}")

    # Combine all summaries into a single context string
    combined_context = " ".join(summaries)

    # Count tokens and truncate if necessary
    tokens = tokenizer.encode(combined_context)
    if len(tokens) > max_tokens:
        truncated_context = tokenizer.decode(tokens[:max_tokens])  # Truncate to fit within max_tokens
    else:
        truncated_context = combined_context

    return truncated_context