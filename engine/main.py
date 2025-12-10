from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse, JSONResponse
from dotenv import load_dotenv
import json
import logging
from agent import FraudAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
load_dotenv()


# Initialize FastAPI app
app = FastAPI(redirect_slashes=False)
fraud_agent = FraudAgent()

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Health check route
@app.get("/")
def read_root():
    logging.info("Health check successful")
    return PlainTextResponse(content="Healthy", status_code=200)

@app.post("/eval")
async def eval(request: Request):
    """
    Run the agent with provided messages and return the tool calls.
    """
    messages = await request.json()

    final_answer, tool_calls = await fraud_agent.aeval(messages)

    return JSONResponse(content={
        "tool_calls": tool_calls,
        "final_answer": final_answer
    })

@app.post("/stream")
async def stream(request: Request):
    """
    Streams back the response from the fraud agent.
    """
    messages = await request.json()
    
    async def generate():
        async for chunk in fraud_agent.ainvoke(messages):
            payload = {"content": chunk}
            # ADD A NEWLINE HERE to separate each JSON chunk
            yield json.dumps(payload) + "\n" 
            
        # The final payload should also have the newline separation
        yield json.dumps({"content": "", "done": True}) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
    )