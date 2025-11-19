from fastapi import FastAPI
from pydantic import BaseModel
from src.employee_support_multiagent import EmployeeSupportOrchestrator

app = FastAPI(title="Employee Support Agent API")

orchestrator = EmployeeSupportOrchestrator()


class QueryRequest(BaseModel):
    session_id: str
    user_id: str
    query: str


@app.post("/query")
def handle_query(req: QueryRequest):
    result = orchestrator.handle_query(
        session_id=req.session_id, user_id=req.user_id, query=req.query
    )
    return result
