# Tests

This directory contains test scripts for the RVC Real-time Voice Conversion application.

## Available Tests

### Validation Test (`validate.py`)

Validates the project structure, Python syntax, and documentation:

```bash
python tests/validate.py
```

This test checks:
- ✓ All required directories exist
- ✓ All required files exist
- ✓ Python syntax is valid for all modules
- ✓ Documentation has required sections

## Running Tests

### Quick Validation
```bash
# From the voice-engine directory
cd services/voice-engine
python tests/validate.py
```

### API Integration Tests

Test the API endpoints (requires running server):

```bash
# Start server first
python main.py --mode api &

# Test WebSocket connection
python examples/websocket_client.py

# Test file conversion
python examples/file_client.py input.wav output.wav
```

### Manual Testing

#### Test API Mode
```bash
# Start the server
python main.py --mode api --log-level DEBUG

# In another terminal, run example client
python examples/websocket_client.py
```

#### Test Docker Build
```bash
# Build the image
docker build -t rvc-real-time .

# Run the container
docker run -p 8765:8765 -p 9876:9876 rvc-real-time
```

#### Test Docker Compose
```bash
docker-compose up
```

## Test Coverage

The validation test ensures:
- Project structure is complete
- All Python files have valid syntax
- Documentation is comprehensive
- Configuration files are present

For full integration testing with dependencies installed, see the main README.md.
