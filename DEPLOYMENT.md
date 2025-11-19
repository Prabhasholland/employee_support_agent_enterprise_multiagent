# Deployment Guide (Cloud Run / Agent Engine style)

## 1. Build and push container

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/employee-support-agent
```

## 2. Deploy to Cloud Run

```bash
gcloud run deploy employee-support-agent \
    --image gcr.io/PROJECT_ID/employee-support-agent \
    --platform managed \
    --region REGION \
    --allow-unauthenticated \
    --port 8080
```

After deployment, your agent API will be available at:

`https://employee-support-agent-<hash>-<region>.run.app/query`

You can POST JSON of the form:

```json
{
  "session_id": "session-123",
  "user_id": "user-001",
  "query": "How many paid vacation days do I get?"
}
```

## 3. Agent Engine / A2A integration (conceptual)

- Wrap the `/query` endpoint as an HTTP tool in Agent Engine.
- Use A2A protocol to connect this employee-support agent to other enterprise agents,
  such as onboarding agents, document-ingestion agents, etc.
