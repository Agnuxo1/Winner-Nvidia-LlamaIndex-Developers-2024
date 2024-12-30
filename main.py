import uvicorn
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
import os

from models.hnn.holographic_neural_network import EnhancedHolographicNeuralNetwork
from models.nemo.nemo_implementation import NemoImplementation
from models.rag.rag_implementation import RAGImplementation

# Inicializar modelos
hnn = EnhancedHolographicNeuralNetwork(dimension=100)
nemo = NemoImplementation()
rag = RAGImplementation()

app = FastAPI(title="Enhanced Holographic Neural Network API")

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory="web/static"), name="static")

class ChatMessage(BaseModel):
    text: str
    type: str = "user"

class LearningInput(BaseModel):
    input_text: str
    response: str

class P2PConnection(BaseModel):
    peer_id: str

@app.post("/api/chat")
async def chat(message: ChatMessage):
    try:
        # Obtener contexto usando RAG
        context = rag.retrieve(message.text)
        
        # Actualizar contexto en HNN
        hnn.update_context_nodes([{
            'text': doc['document']['text'],
            'score': doc['score']
        } for doc in context])
        
        # Generar respuesta usando NeMo
        nemo_response = nemo.generate_response(message.text)
        
        # Aplicar guardrails
        safe_response = nemo.apply_guardrails(nemo_response)
        
        # Aprender de la interacción
        hnn.learn(message.text, safe_response)
        
        return JSONResponse({
            "response": safe_response,
            "context": context[:3]  # Devolver top 3 contextos
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/api/learn")
async def learn(input_data: LearningInput):
    try:
        hnn.learn(input_data.input_text, input_data.response)
        return {"status": "success"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Procesar archivo
        content = await file.read()
        
        if file.filename.endswith('.pdf'):
            # Procesar PDF
            # Implementar procesamiento de PDF
            pass
        elif file.filename.endswith('.txt'):
            # Procesar texto
            text = content.decode()
            rag.add_documents([{
                'text': text,
                'metadata': {'filename': file.filename}
            }])
            
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['type'] == 'chat':
                # Procesar mensaje de chat
                response = await chat(ChatMessage(text=message['text']))
                await websocket.send_json(response)
            elif message['type'] == 'p2p':
                # Manejar comunicación P2P
                # Implementar lógica P2P
                pass
                
    except Exception as e:
        await websocket.close()

if __name__ == "__main__":
    # Crear directorios necesarios
    os.makedirs("web/static", exist_ok=True)
    
    # Iniciar servidor
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
