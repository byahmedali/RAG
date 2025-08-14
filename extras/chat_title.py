# Function to generate chat title from user query
def generate_chat_title(query):
    """Generate a meaningful chat title from the user's first query"""
    # Clean and truncate the query
    clean_query = query.strip()

    # If query is short enough, use it directly
    if len(clean_query) <= 40:
        return clean_query

    # For longer queries, try to extract the main topic
    # Look for question words and key terms
    question_words = ["what", "how", "why", "when", "where", "which", "who"]
    words = clean_query.lower().split()

    # Find the first question word or key term
    for i, word in enumerate(words):
        if word in question_words or len(word) > 5:  # Longer words are likely key terms
            # Take words from this point, up to 40 characters
            title_words = words[i : i + 8]  # Take up to 8 words
            title = " ".join(title_words)
            if len(title) > 40:
                title = title[:37] + "..."
            return title.capitalize()

    # Fallback: take first 40 characters
    return clean_query[:37] + "..."


if __name__ == "__main__":
    generate_chat_title()
