from collections import Counter


class SearchRanker:
    def rank_by_word_frequency(self, query_keywords, paragraphs):
        # Tokenize the query and text chunks
        query_tokens = " ".join(query_keywords).lower().split(" ")
        text_chunks_tokens = [chunk.lower().replace(",", " ").split(" ") for chunk in paragraphs]

        # Calculate word frequencies for the query
        query_word_frequencies = Counter(query_tokens)

        # print(query_word_frequencies)

        # Calculate word frequencies for each text chunk
        text_chunks_word_frequencies = [Counter(chunk_tokens) for chunk_tokens in text_chunks_tokens]
        for i in range(len(text_chunks_word_frequencies)):
            for key in text_chunks_word_frequencies[i]:
                text_chunks_word_frequencies[i][key] = 1

        # Calculate the relevance score for each text chunk based on word frequencies
        relevance_scores = [
            sum((query_word_frequencies & chunk_freq).values()) for chunk_freq in text_chunks_word_frequencies
        ]

        # Create a list of (index, relevance score) tuples
        ranked_chunks = [(i, score) for i, score in enumerate(relevance_scores)]

        # Sort the list based on the relevance scores in descending order
        ranked_chunks.sort(key=lambda x: x[1], reverse=True)

        # Extract the indices of the sorted chunks
        sorted_indices = [index for index, _ in ranked_chunks]

        # Return the sorted text chunks
        return [paragraphs[i] for i in sorted_indices]
