import heapq
import numpy as np
import ollama

def handle_user_input():
    """Function to handle user input."""
    if prompt := input("Enter your query:"):
        return prompt

def cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

def plot_vectors(embedding1, embedding2):
    """Plot the two vectors in 2D space with dynamic axis."""
    plt.figure(figsize=(6, 6))
    
    # Origin for vectors
    origin = np.array([0, 0])

    # Plot vectors
    plt.quiver(*origin, *embedding1, angles='xy', scale_units='xy', scale=1, color='r', label='Vector 1')
    plt.quiver(*origin, *embedding2, angles='xy', scale_units='xy', scale=1, color='b', label='Vector 2')

    # Calculate axis limits based on the vector values
    all_vectors = np.vstack([embedding1, embedding2])
    max_range = np.max(np.abs(all_vectors))
    
    # Set dynamic axis limits
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)

    # Add fine gridlines
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Fine gridlines with dashed style
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    # Add major and minor ticks for finer control over gridlines
    plt.minorticks_on()  # Enable minor ticks
    plt.tick_params(which='minor', length=3, width=0.5, colors='black', grid_color='gray', grid_alpha=0.5)  # Minor ticks

    # Label the axes
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    # Display cosine similarity on the plot
    similarity = cosine_similarity(embedding1, embedding2)
    plt.title(f'Cosine Similarity: {similarity:.2f}')
    
    plt.show()


def get_user_input():
    """Get user input for two vectors in 2D space."""
    print("\nPlease enter the coordinates for two vectors in a 2D space.")

    # Initialize an empty list to store the embeddings
    embeddings = []

    # Loop to get input for both vectors
    for i in range(1, 3):
        print(f"\nFor Vector {i}, please provide the coordinates.")
        x = float(input(f"Enter the x-coordinate of Vector {i}: "))
        y = float(input(f"Enter the y-coordinate of Vector {i}: "))
        embeddings.append(np.array([x, y]))

    # Return the two points as vectors
    return embeddings[0], embeddings[1]


def main():
    """Main function to perform semantic search in a loop."""
    while True:
        print("\n" + "="*50)
        print("Semantic Search")
        print("="*50)

        user_input = handle_user_input()
        if user_input:
            # Perform search and display relevant documents
            relevant_documents = search(user_input, documents, document_embeddings)
            print(f"\nRelevant Documents for your query: '{user_input}'\n")
            print("-"*50)
            print(f"{relevant_documents}")
            print("-"*50)

        continue_query = input("\nDo you want to search again? (y/Y to continue, any other key to quit): ")
        if continue_query.lower() != "y":
            print("\nExiting!")
            break


def embed_text(text: str, embedding_model="nomic-embed-text"):
    """Embed text using Ollama's specified embedding model (default: nomic-embed-text)."""
    response = ollama.embeddings(model=embedding_model, prompt=text)
    embedding = response["embedding"]
    return np.array(embedding)

def topk(arr, k):
    """Get the k largest elements and their indexes."""
    return heapq.nlargest(k, range(len(arr)), key=lambda i: arr[i])

def search(query, documents, document_embeddings, top_k=5):
    """Search for the most relevant document based on the cosine similarity of embeddings."""
    query_embedding = embed_text(query)

    similarities = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in document_embeddings]

    topk_indices = topk(similarities, top_k)
    results = []
    for i in topk_indices:
        #logging.info(f"Document: {documents[i]}\nSimilarity Score: {similarities[i]}\n")
        results.append(f"Document: {documents[i]}\nSimilarity Score: {similarities[i]:.4f}")

    result = '\n\n'.join(results)
    return result

if __name__ == "__main__":
    documents = [
        # hotel_details
        "The Grand Palace Hotel offers luxury rooms with a view of the city skyline.",
        "Our hotel is located in the heart of downtown, just a 10-minute walk from the main shopping district.",
        "Each room comes equipped with a flat-screen TV, free Wi-Fi, and premium toiletries.",
        "The hotel features a rooftop pool, fitness center, and spa services for all guests.",
        "Check-in time starts at 3:00 PM and check-out is at 11:00 AM.",
        # booking_policy
        "To secure your reservation, a credit card is required at the time of booking.",
        "Rooms can be booked up to six months in advance on our website.",
        "All reservations must be guaranteed with a valid payment method.",
        "The hotel accepts bookings for a minimum stay of two nights during peak seasons.",
        "If you'd like to modify your booking, please contact the front desk at least 24 hours before your check-in date.",
        # cancellation_policy
        "Guests can cancel their reservation free of charge up to 48 hours before the scheduled check-in date.",
        "Cancellations made within 24 hours of check-in will incur a 50% charge of the total booking cost.",
        "No-show guests will be charged the full amount of their stay.",
        "To cancel your reservation, you can either call the hotel directly or cancel through our website.",
        # refund_policy
        "Refunds are processed to the original payment method within 7 business days.",
        "Refunds are only available for cancellations made within the allowed time frame, as per the hotel's cancellation policy.",
        "If a guest has prepaid for their stay and cancels within the eligible period, a full refund will be issued.",
        "Refunds for partial bookings will be calculated based on the number of nights stayed and the cancellation policy.",
        # food_and_dining
        "Our hotel offers complimentary breakfast every morning from 7:00 AM to 10:00 AM.",
        "The on-site restaurant serves a range of international dishes, with vegetarian and vegan options available.",
        "Room service is available 24/7 for all guests.",
        "Guests can enjoy afternoon tea in the lobby lounge from 3:00 PM to 5:00 PM.",
        "The hotel bar serves a variety of cocktails, wine, and snacks from 12:00 PM to midnight.",
        # additional_amenities
        "Free parking is available for all guests staying at the hotel.",
        "We offer airport shuttle services for an additional fee. Please book in advance.",
        "The hotel provides laundry services, including dry cleaning and pressing.",
        "There is a business center on-site, with printing and copying services available for guests."
    ]

    document_embeddings = [embed_text(doc) for doc in documents]
    main()
