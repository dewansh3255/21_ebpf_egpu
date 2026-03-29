import torch
import torch.nn as nn
from torchvision.models import resnet18
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import uvicorn
import io

# 1. Initialize the FastAPI Web Server
app = FastAPI(title="Federated Learning Global Server")

class FederatedServer:
    def __init__(self):
        self.device = torch.device("cpu") 
        self.global_model = resnet18(num_classes=10)
        self.global_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.global_model.maxpool = nn.Identity()
        self.global_model.to(self.device)
        
        # Network tracking
        self.expected_clients = 2
        self.received_weights = []
        self.current_round = 1

    def aggregate_weights(self):
        print(f"\n[Server] All {self.expected_clients} clients received. Running FedAvg for Round {self.current_round}...")
        global_dict = self.global_model.state_dict()
        
        for key in global_dict.keys():
            temp_tensor = torch.zeros_like(global_dict[key], dtype=torch.float32)
            for client_dict in self.received_weights:
                temp_tensor += client_dict[key]
            temp_tensor = temp_tensor / len(self.received_weights)
            global_dict[key] = temp_tensor
            
        self.global_model.load_state_dict(global_dict)
        print(f"[Server] Round {self.current_round} complete! Global model updated.\n")
        
        # Reset for the next round
        self.received_weights = []
        self.current_round += 1

# Instantiate our server logic
fl_server = FederatedServer()

# --- API ENDPOINTS ---

@app.get("/get_weights")
async def get_weights():
    """Clients call this to download the current global master weights."""
    buffer = io.BytesIO()
    torch.save(fl_server.global_model.state_dict(), buffer)
    buffer.seek(0)
    return Response(content=buffer.read(), media_type="application/octet-stream")

@app.post("/upload_weights")
async def upload_weights(file: UploadFile = File(...)):
    """Clients post their trained weights here."""
    contents = await file.read()
    buffer = io.BytesIO(contents)
    client_state_dict = torch.load(buffer, map_location="cpu", weights_only=True)
    
    fl_server.received_weights.append(client_state_dict)
    print(f"[Network] Received weights from a client. ({len(fl_server.received_weights)}/{fl_server.expected_clients})")
    
    # If we have collected weights from all clients, run the FedAvg math!
    if len(fl_server.received_weights) == fl_server.expected_clients:
        fl_server.aggregate_weights()
        
    return {"status": "Weights received successfully"}

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    print("Starting Federated API Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)