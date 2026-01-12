# WSL2 Port Forwarding Script for MorphVox
# Run this script as Administrator whenever WSL IP changes or on Windows startup
#
# To run: Right-click > Run with PowerShell (as Admin)
# Or from elevated PowerShell: .\wsl_port_forward.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== MorphVox WSL2 Port Forwarding Setup ===" -ForegroundColor Cyan

# Get the WSL2 IP address
$wslIP = (wsl hostname -I).Trim().Split(" ")[0]

if ([string]::IsNullOrEmpty($wslIP)) {
    Write-Host "ERROR: Could not get WSL2 IP. Is WSL running?" -ForegroundColor Red
    exit 1
}

Write-Host "WSL2 IP Address: $wslIP" -ForegroundColor Green

# Ports to forward
$ports = @(80, 443)

# Reset existing port proxy rules
Write-Host "`nResetting existing port proxy rules..." -ForegroundColor Yellow
netsh interface portproxy reset

# Add new port proxy rules
foreach ($port in $ports) {
    Write-Host "Forwarding port $port to WSL2..." -ForegroundColor Yellow
    netsh interface portproxy add v4tov4 listenport=$port listenaddress=0.0.0.0 connectport=$port connectaddress=$wslIP
}

# Show current rules
Write-Host "`nCurrent port proxy rules:" -ForegroundColor Cyan
netsh interface portproxy show all

# Ensure Windows Firewall allows the ports
Write-Host "`nConfiguring Windows Firewall..." -ForegroundColor Yellow

foreach ($port in $ports) {
    $ruleName = "WSL2 Port $port"
    $existingRule = netsh advfirewall firewall show rule name="$ruleName" 2>$null
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Adding firewall rule for port $port..." -ForegroundColor Yellow
        netsh advfirewall firewall add rule name="$ruleName" dir=in action=allow protocol=tcp localport=$port
    } else {
        Write-Host "Firewall rule for port $port already exists" -ForegroundColor Gray
    }
}

Write-Host "`n=== Port forwarding configured successfully! ===" -ForegroundColor Green
Write-Host "WSL2 IP: $wslIP"
Write-Host "Forwarded ports: $($ports -join ', ')"
Write-Host "`nYou can now access MorphVox from the internet." -ForegroundColor Cyan

# Optional: Test connectivity
Write-Host "`nTesting local connectivity..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost" -TimeoutSec 5 -UseBasicParsing
    Write-Host "HTTP test: OK (Status $($response.StatusCode))" -ForegroundColor Green
} catch {
    Write-Host "HTTP test: Could not connect (containers may not be running)" -ForegroundColor Yellow
}

Read-Host "`nPress Enter to exit"
