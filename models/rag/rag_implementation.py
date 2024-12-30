from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class RAGImplementation:
    def __init__(self, embedding_dim: int = 768):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        
        # Inicializar modelos
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(self.device)
        
        # Inicializar índice FAISS
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Inicializar vectorizador TF-IDF para búsqueda híbrida
        self.tfidf = TfidfVectorizer()
        
        # Base de conocimiento
        self.documents: List[Dict[str, Any]] = []

    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Añade documentos a la base de conocimiento.
        
        Args:
            documents: Lista de documentos con campos 'text' y 'metadata'
        """
        for doc in documents:
            # Generar embedding
            embedding = self._generate_embedding(doc['text'])
            
            # Añadir al índice FAISS
            self.index.add(np.array([embedding]))
            
            # Almacenar documento
            self.documents.append({
                'text': doc['text'],
                'metadata': doc.get('metadata', {}),
                'embedding': embedding.tolist()
            })
        
        # Actualizar TF-IDF
        texts = [doc['text'] for doc in self.documents]
        self.tfidf.fit(texts)

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Genera embedding para un texto."""
        inputs = self.tokenizer(text, return_tensors="pt", 
                              truncation=True, max_length=512,
                              padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings[0].cpu().numpy()

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Recupera los documentos más relevantes para una consulta.
        
        Args:
            query: Texto de la consulta
            k: Número de documentos a recuperar
            
        Returns:
            Lista de documentos relevantes con scores
        """
        # Generar embedding de la consulta
        query_embedding = self._generate_embedding(query)
        
        # Búsqueda por similitud de embeddings
        D, I = self.index.search(np.array([query_embedding]), k)
        
        # Búsqueda híbrida con TF-IDF
        tfidf_scores = self.tfidf.transform([query])
        
        results = []
        for i, (distance, idx) in enumerate(zip(D[0], I[0])):
            if idx < len(self.documents):
                doc = self.documents[idx]
                
                # Calcular score híbrido
                tfidf_score = tfidf_scores.dot(
                    self.tfidf.transform([doc['text']]).T
                )[0, 0]
                
                # Normalizar scores
                embedding_score = 1 / (1 + distance)
                final_score = 0.7 * embedding_score + 0.3 * tfidf_score
                
                results.append({
                    'document': doc,
                    'score': float(final_score)
                })
        
        # Ordenar por score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def save_index(self, path: str):
        """Guarda el índice y la base de conocimiento."""
        try:
            # Guardar índice FAISS
            faiss.write_index(self.index, f"{path}_faiss.index")
            
            # Guardar documentos
            import json
            with open(f"{path}_docs.json", 'w') as f:
                json.dump(self.documents, f)
                
        except Exception as e:
            print(f"Error saving index: {e}")

    def load_index(self, path: str):
        """Carga el índice y la base de conocimiento."""
        try:
            # Cargar índice FAISS
            self.index = faiss.read_index(f"{path}_faiss.index")
            
            # Cargar documentos
            import json
            with open(f"{path}_docs.json", 'r') as f:
                self.documents = json.load(f)
                
            # Actualizar TF-IDF
            texts = [doc['text'] for doc in self.documents]
            self.tfidf.fit(texts)
            
        except Exception as e:
            print(f"Error loading index: {e}")

    def update_document(self, doc_id: int, new_text: str):
        """Actualiza un documento existente."""
        if 0 <= doc_id < len(self.documents):
            # Generar nuevo embedding
            new_embedding = self._generate_embedding(new_text)
            
            # Actualizar documento
            self.documents[doc_id]['text'] = new_text
            self.documents[doc_id]['embedding'] = new_embedding.tolist()
            
            # Reconstruir índice FAISS
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            embeddings = [np.array(doc['embedding']) for doc in self.documents]
            if embeddings:
                self.index.add(np.vstack(embeddings))
            
            # Actualizar TF-IDF
            texts = [doc['text'] for doc in self.documents]
            self.tfidf.fit(texts)
