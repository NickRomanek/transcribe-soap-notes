# AI Scribe Docker Runner for Windows
# Automatically detects GPU and runs appropriate configuration

param(
    [string]$Mode = "menu"
)

# Colors for output
$Red = "Red"
$Green = "Green"
$Yellow = "Yellow"
$Blue = "Blue"
$White = "White"

function Write-ColorOutput($Message, $Color) {
    Write-Host $Message -ForegroundColor $Color
}

Write-ColorOutput "🚀 AI Scribe Docker Setup" $Blue
Write-ColorOutput "========================================" $Blue

# Check if .env file exists
if (!(Test-Path ".env")) {
    Write-ColorOutput "⚠️  No .env file found. Creating from template..." $Yellow
    Copy-Item "env.sample" ".env"
    Write-ColorOutput "📝 Please edit .env file with your HuggingFace token" $Yellow
    Write-ColorOutput "   Get token from: https://huggingface.co/settings/tokens" $Yellow
    Read-Host "Press Enter to continue once you've configured .env"
}

# Check for NVIDIA Docker support
$GPU_AVAILABLE = $false
try {
    $dockerInfo = docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "✅ NVIDIA Container Runtime detected" $Green
        $GPU_AVAILABLE = $true
    }
}
catch {
    Write-ColorOutput "⚠️  No GPU support detected, will run on CPU" $Yellow
}

# Create directories
Write-ColorOutput "📁 Creating directories..." $Blue
if (!(Test-Path "data")) { New-Item -ItemType Directory -Path "data" | Out-Null }
if (!(Test-Path "models")) { New-Item -ItemType Directory -Path "models" | Out-Null }
if (!(Test-Path "output")) { New-Item -ItemType Directory -Path "output" | Out-Null }

# Function to display menu
function Show-Menu {
    Write-Host ""
    Write-ColorOutput "Choose deployment option:" $Blue
    Write-Host "1) API Server (REST API)"
    Write-Host "2) Web Interface (Gradio)"
    Write-Host "3) CLI Interface (Interactive)"
    Write-Host "4) Download Models Only"
    Write-Host "5) Development Mode (with Jupyter)"
    Write-Host "6) CPU-only Mode"
    Write-Host "7) Stop All Services"
    Write-Host "8) Show Logs"
    Write-Host "q) Quit"
    Write-Host ""
}

# Function to start services
function Start-Service($Profile, $ServiceName) {
    Write-ColorOutput "🚀 Starting $ServiceName..." $Blue
    
    if ($GPU_AVAILABLE -and $Profile -ne "cpu") {
        docker-compose --profile $Profile up -d
    } else {
        docker-compose --profile cpu up -d
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "✅ $ServiceName started successfully!" $Green
        
        # Show relevant URLs
        switch ($Profile) {
            { $_ -in "api", "default" } {
                Write-ColorOutput "📡 API Server: http://localhost:8000" $Green
                Write-ColorOutput "📚 API Docs: http://localhost:8000/docs" $Green
            }
            "web" {
                Write-ColorOutput "🌐 Web Interface: http://localhost:7860" $Green
            }
            "dev" {
                Write-ColorOutput "📡 API Server: http://localhost:8000" $Green
                Write-ColorOutput "📓 Jupyter Lab: http://localhost:8888" $Green
            }
        }
    } else {
        Write-ColorOutput "❌ Failed to start $ServiceName" $Red
    }
}

# Function to stop services
function Stop-Services {
    Write-ColorOutput "🛑 Stopping all services..." $Yellow
    docker-compose down
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "✅ All services stopped" $Green
    }
}

# Function to show logs
function Show-Logs {
    Write-ColorOutput "📋 Recent logs:" $Blue
    docker-compose logs --tail=20
}

# Main menu loop
while ($true) {
    Show-Menu
    $choice = Read-Host "Enter your choice"
    
    switch ($choice) {
        "1" {
            Start-Service "api" "API Server"
        }
        "2" {
            Start-Service "web" "Web Interface"
        }
        "3" {
            Write-ColorOutput "🔧 Starting CLI interface..." $Blue
            docker-compose run --rm ai-scribe-cli bash
        }
        "4" {
            Write-ColorOutput "📥 Downloading models..." $Blue
            docker-compose --profile download up ai-scribe-download
        }
        "5" {
            Start-Service "dev" "Development Mode"
        }
        "6" {
            Start-Service "cpu" "CPU-only Mode"
        }
        "7" {
            Stop-Services
        }
        "8" {
            Show-Logs
        }
        { $_ -in "q", "Q" } {
            Write-ColorOutput "👋 Goodbye!" $Blue
            break
        }
        default {
            Write-ColorOutput "❌ Invalid option. Please try again." $Red
        }
    }
    
    if ($choice -notin @("7", "8", "q", "Q")) {
        Write-Host ""
        Write-ColorOutput "Press Enter to continue..." $Blue
        Read-Host
    }
} 