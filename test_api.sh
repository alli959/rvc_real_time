#!/bin/bash
# Test if API returns model paths

echo "Testing API response for model paths..."
echo ""

TOKEN=$(docker exec morphvox-api sh -c "cd /var/www/html && php artisan tinker" <<'EOF'
$user = \App\Models\User::first();
$token = $user->createToken('test')->plainTextToken;
echo $token;
EOF
)

echo "Got token (truncated): ${TOKEN:0:20}..."
echo ""

# Test with authentication
docker exec morphvox-api sh -c "wget -qO- --header='Authorization: Bearer ${TOKEN}' --header='Accept: application/json' http://localhost/api/voice-models" 2>&1 | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'data' in data and len(data['data']) > 0:
        model = data['data'][0]
        print(f\"Model name: {model.get('name')}\")
        print(f\"Has model_path: {'model_path' in model}\")
        print(f\"Has index_path: {'index_path' in model}\")
        if 'model_path' in model:
            print(f\"model_path value: {model['model_path']}\")
        if 'index_path' in model:
            print(f\"index_path value: {model['index_path']}\")
    else:
        print('No data returned')
        print(data)
except Exception as e:
    print(f'Error: {e}')
    print('Raw stdin:', sys.stdin.read())
"
