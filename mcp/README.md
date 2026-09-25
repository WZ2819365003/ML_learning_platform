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

## Tianyi Cloud Deployment Note

- The Tianyi Cloud host reserves public HTTP ports `18081` through `18090` for
  web applications.
- `ML_platform` is deployed on public port `18081`.
- The production entrypoint is nginx, mapped as
  `${PUBLIC_HTTP_PORT:-18081}:80` in `docker/docker-compose.yml`.
- Backend, frontend, MySQL, Redis, and MinIO stay bound to loopback on the
  server and should not be accessed directly from the public network.
