from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents
documents = ["This is the first document.",
             "This document is the second document.",
             "And this is the third one."]

# Create an instance of CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the documents into a sparse matrix
X = vectorizer.fit_transform(documents)

# Convert the sparse matrix to a dense array for better visibility
dense_array = X.toarray()

# Print the vocabulary and the document-term matrix
print("Vocabulary:")
print(vectorizer.get_feature_names_out())
print("\nDocument-Term Matrix:")
print(dense_array)

text = ['nikki is girl', 'nikki love is rujan']
textVectorized = vectorizer.fit_transform(text)
print(textVectorized)

cosine_sim = cosine_similarity(textVectorized[0], textVectorized[1])
print(cosine_sim)


