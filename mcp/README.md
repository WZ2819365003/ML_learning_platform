# Local MCP: Doubao ML Report

This folder contains the local MCP wrapper for Doubao/Volcengine Ark report
generation.

Configure the API key in your shell or local MCP environment:

```bash
export ARK_API_KEY="your-ark-api-key"
```

Example MCP server config:

```json
{
  "mcpServers": {
    "doubao-report": {
      "command": "python",
      "args": ["mcp/doubao_report_mcp.py"],
      "env": {
        "ARK_API_KEY": "${ARK_API_KEY}"
      }
    }
  }
}
```

Tools:

- `check_doubao_reachability`: calls the chat completion endpoint with a small
  connectivity prompt.
- `generate_ml_report`: accepts a task summary object/string and returns a
  Chinese Markdown report with conclusion, explanation, and suggestions.
