import spacy

nlp = spacy.load("en_core_web_sm")

def semantic_chunk_text(text, max_words=40):
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in doc.sents:
        words = sent.text.split()
        if current_len + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0

        current_chunk.append(sent.text)
        current_len += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    json_chunks  = {
        "total_chunks" : len(chunks),
        "chunks" : []
    }
    for i,chunk in enumerate(chunks):
        json_chunks["chunks"].append({
            "chunk_id" : i,
            "text" : chunk.strip()
        })

    return json_chunks
