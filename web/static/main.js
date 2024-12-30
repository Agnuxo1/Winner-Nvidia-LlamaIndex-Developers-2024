// WebSocket connection
let ws = null;

// Initialize WebSocket connection
function initWebSocket() {
    ws = new WebSocket(`ws://${window.location.host}/ws`);
    
    ws.onmessage = function(event) {
        const response = JSON.parse(event.data);
        displayMessage(response.response, 'bot');
        
        // Display context if available
        if (response.context) {
            response.context.forEach(ctx => {
                displayContext(ctx);
            });
        }
    };

    ws.onclose = function() {
        // Attempt to reconnect after 2 seconds
        setTimeout(initWebSocket, 2000);
    };
}

// Display a message in the chat
function displayMessage(text, type) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type} mb-4 p-3 rounded-lg ${
        type === 'user' ? 'bg-blue-100 ml-12' : 'bg-gray-100 mr-12'
    }`;
    messageDiv.textContent = text;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Display context information
function displayContext(context) {
    const chatMessages = document.getElementById('chat-messages');
    const contextDiv = document.createElement('div');
    contextDiv.className = 'context text-xs text-gray-500 mb-2 ml-4';
    contextDiv.textContent = `Context (${context.score.toFixed(2)}): ${context.document.text.substring(0, 100)}...`;
    chatMessages.appendChild(contextDiv);
}

// Send a chat message
function sendMessage() {
    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    
    if (text && ws) {
        displayMessage(text, 'user');
        ws.send(JSON.stringify({
            type: 'chat',
            text: text
        }));
        input.value = '';
    }
}

// Handle file upload
async function handleFileUpload() {
    const fileInput = document.getElementById('file-upload');
    const file = fileInput.files[0];
    
    if (file) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            if (result.status === 'success') {
                alert('File uploaded successfully!');
            } else {
                alert('Error uploading file: ' + result.error);
            }
        } catch (error) {
            alert('Error uploading file: ' + error);
        }
    }
}

// Submit learning data
async function submitLearning() {
    const input = document.getElementById('learn-input').value.trim();
    const response = document.getElementById('learn-response').value.trim();
    
    if (input && response) {
        try {
            const result = await fetch('/api/learn', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    input_text: input,
                    response: response
                })
            });
            
            const data = await result.json();
            if (data.status === 'success') {
                alert('Learning submitted successfully!');
                document.getElementById('learn-input').value = '';
                document.getElementById('learn-response').value = '';
            } else {
                alert('Error submitting learning: ' + data.error);
            }
        } catch (error) {
            alert('Error submitting learning: ' + error);
        }
    }
}

// Handle Enter key in chat input
document.getElementById('chat-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Initialize WebSocket when page loads
window.onload = function() {
    initWebSocket();
};
