#!/bin/bash
# Setup virtual audio devices for RVC on Linux (PulseAudio/PipeWire)
#
# This creates:
#   - RVC_Sink: A null sink where converted audio goes
#   - RVC_Mic: A virtual microphone that mirrors RVC_Sink
#
# After running this, select "RVC_Mic" as your microphone in Discord/Zoom/etc.

set -e

echo "=== RVC Virtual Microphone Setup ==="
echo

# Check if PulseAudio or PipeWire is running
if command -v pactl &> /dev/null; then
    echo "Found PulseAudio/PipeWire"
else
    echo "Error: pactl not found. Please install PulseAudio or PipeWire."
    exit 1
fi

# Remove existing modules if they exist (ignore errors)
echo "Cleaning up existing virtual devices..."
pactl unload-module module-null-sink 2>/dev/null || true
pactl unload-module module-virtual-source 2>/dev/null || true

# Create the null sink (this is where RVC outputs to)
echo "Creating RVC_Sink (output device)..."
pactl load-module module-null-sink \
    sink_name=RVC_Sink \
    sink_properties=device.description="RVC_Output"

# Create the virtual microphone (this monitors the sink)
echo "Creating RVC_Mic (virtual microphone)..."
pactl load-module module-virtual-source \
    source_name=RVC_Mic \
    master=RVC_Sink.monitor \
    source_properties=device.description="RVC_Microphone"

echo
echo "=== Setup Complete ==="
echo
echo "Virtual devices created:"
echo "  - RVC_Sink: Use this as output in virtual_mic_client.py"
echo "  - RVC_Mic: Select this as your microphone in Discord/Zoom/etc."
echo
echo "Usage:"
echo "  1. Start RVC server:"
echo "     python3 main.py --mode api --model ./assets/models/BillCipher/BillCipher.pth --index ./assets/models/BillCipher/BillCipher.index"
echo
echo "  2. Start virtual mic client:"
echo "     python3 examples/virtual_mic_client.py --output-device RVC_Sink"
echo
echo "  3. In Discord/Zoom/etc, select 'RVC_Mic' as your microphone"
echo
echo "Note: These devices are temporary and will be removed on reboot."
echo "Run this script again after reboot, or add to startup."
