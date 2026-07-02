# Migrating From Boto3 Bedrock Runtime

Bedmock is intended for applications that already have Bedrock Runtime-shaped inference code and
want to route that code to an OpenAI-compatible provider.

## Install

```bash
python -m pip install "git+https://github.com/alexkrasov/bedmock.git"
```

## Minimal Import Change

For the smallest migration, change only the import:

```python
# import boto3
import bedmock as boto3

client = boto3.client("bedrock-runtime")
```

The existing Bedrock-shaped call can keep using `invoke_model`, `invoke_model_with_response_stream`,
`converse`, `converse_stream`, or `count_tokens` when the source payload shape is supported.

## Provider Configuration

Bedmock does not load `.env` files automatically. Either export the needed variables in your shell
or load them in your application before creating the client.

```bash
export BEDROCK_BRIDGE_PROVIDER=gemini
export BEDROCK_BRIDGE_MODEL="<current-gemini-model>"
export GEMINI_API_KEY="<secret>"
```

Then run a non-billing diagnostic check:

```bash
bedmock doctor --model-id "${BEDROCK_MODEL_ID:-us.amazon.nova-2-lite-v1:0}"
```

## Mixed AWS Applications

If your code uses other AWS services, prefer an explicit import boundary:

```python
import boto3
import bedmock

s3 = boto3.client("s3")
llm = bedmock.client("bedrock-runtime")
```

If you need a one-line drop-in import for both Bedrock Runtime and other AWS services, install the
AWS extra and enable delegation:

```bash
python -m pip install "bedmock[aws] @ git+https://github.com/alexkrasov/bedmock.git"
export BEDROCK_BRIDGE_DELEGATE_OTHER_SERVICES=true
```

## Legacy Namespace

The earlier `bedrock_bridge` import namespace remains available:

```python
import bedrock_bridge as boto3
```

New code should prefer `bedmock`.
