# Compatibility

## Boto3 Facade

The following imports work:

```python
import bedmock as boto3
import bedrock_bridge as boto3
from bedmock import client
from bedmock import Session
from bedrock_bridge import client
from bedrock_bridge import Session
from bedrock_bridge.session import Session
```

Supported client creation:

```python
boto3.client("bedrock-runtime")
boto3.Session(...).client("bedrock-runtime")
boto3.session.Session(...).client("bedrock-runtime")
```

Common boto3 client parameters are accepted for compatibility. AWS credentials are stored only as
metadata and are not sent to external LLM providers.

## Mixed AWS Applications

For code that uses both Bedrock Runtime inference and other AWS services, prefer an explicit import
boundary:

```python
import boto3
import bedmock

s3 = boto3.client("s3")
ec2 = boto3.client("ec2")
llm = bedmock.client("bedrock-runtime")
```

If the application needs the one-line drop-in import, install the AWS extra and enable delegation:

```bash
python -m pip install "bedmock[aws] @ git+https://github.com/alexkrasov/bedmock.git"
export BEDROCK_BRIDGE_DELEGATE_OTHER_SERVICES=true
```

Then `import bedmock as boto3` keeps `bedrock-runtime` on Bedmock and sends other service clients
to the installed `boto3` package.

## Response Compatibility

`invoke_model` returns a `botocore.response.StreamingBody`, so existing code can call:

```python
payload = json.loads(response["body"].read())
```

Converse responses expose `output.message.content`, `stopReason`, `usage`, and `ResponseMetadata`.

## Unsupported Operations

The following are stable diagnostic stubs:

- `apply_guardrail`
- `start_async_invoke`
- `get_async_invoke`
- `list_async_invokes`

They raise `UnsupportedOperationException`, not `AttributeError`.
