#!/bin/bash

# AI Scribe Docker Runner
# Automatically detects GPU and runs appropriate configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 AI Scribe Docker Setup${NC}"
echo "========================================"

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  No .env file found. Creating from template...${NC}"
    cp env.sample .env
    echo -e "${YELLOW}📝 Please edit .env file with your HuggingFace token${NC}"
    echo -e "${YELLOW}   Get token from: https://huggingface.co/settings/tokens${NC}"
    read -p "Press enter to continue once you've configured .env..."
fi

# Check for NVIDIA Docker support
if command -v nvidia-docker &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA Docker detected${NC}"
    GPU_AVAILABLE=true
elif docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA Container Runtime detected${NC}"
    GPU_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  No GPU support detected, will run on CPU${NC}"
    GPU_AVAILABLE=false
fi

# Create directories
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p data models output

# Function to display menu
show_menu() {
    echo ""
    echo -e "${BLUE}Choose deployment option:${NC}"
    echo "1) API Server (REST API)"
    echo "2) Web Interface (Gradio)"
    echo "3) CLI Interface (Interactive)"
    echo "4) Download Models Only"
    echo "5) Development Mode (with Jupyter)"
    echo "6) CPU-only Mode"
    echo "7) Stop All Services"
    echo "q) Quit"
    echo ""
}

# Function to start services
start_service() {
    local profile=$1
    local service_name=$2
    
    echo -e "${BLUE}🚀 Starting ${service_name}...${NC}"
    
    if [ "$GPU_AVAILABLE" = true ] && [ "$profile" != "cpu" ]; then
        docker-compose --profile $profile up -d
    else
        docker-compose --profile cpu up -d
    fi
    
    echo -e "${GREEN}✅ ${service_name} started successfully!${NC}"
    
    # Show relevant URLs
    case $profile in
        "api"|"default")
            echo -e "${GREEN}📡 API Server: http://localhost:8000${NC}"
            echo -e "${GREEN}📚 API Docs: http://localhost:8000/docs${NC}"
            ;;
        "web")
            echo -e "${GREEN}🌐 Web Interface: http://localhost:7860${NC}"
            ;;
        "dev")
            echo -e "${GREEN}📡 API Server: http://localhost:8000${NC}"
            echo -e "${GREEN}📓 Jupyter Lab: http://localhost:8888${NC}"
            ;;
    esac
}

# Function to stop services
stop_services() {
    echo -e "${YELLOW}🛑 Stopping all services...${NC}"
    docker-compose down
    echo -e "${GREEN}✅ All services stopped${NC}"
}

# Function to show logs
show_logs() {
    echo -e "${BLUE}📋 Recent logs:${NC}"
    docker-compose logs --tail=20
}

# Main menu loop
while true; do
    show_menu
    read -p "Enter your choice: " choice
    
    case $choice in
        1)
            start_service "api" "API Server"
            ;;
        2)
            start_service "web" "Web Interface"
            ;;
        3)
            echo -e "${BLUE}🔧 Starting CLI interface...${NC}"
            docker-compose run --rm ai-scribe-cli bash
            ;;
        4)
            echo -e "${BLUE}📥 Downloading models...${NC}"
            docker-compose --profile download up ai-scribe-download
            ;;
        5)
            start_service "dev" "Development Mode"
            ;;
        6)
            start_service "cpu" "CPU-only Mode"
            ;;
        7)
            stop_services
            ;;
        q|Q)
            echo -e "${BLUE}👋 Goodbye!${NC}"
            break
            ;;
        *)
            echo -e "${RED}❌ Invalid option. Please try again.${NC}"
            ;;
    esac
    
    if [ "$choice" != "7" ] && [ "$choice" != "q" ] && [ "$choice" != "Q" ]; then
        echo ""
        echo -e "${BLUE}Press Enter to continue...${NC}"
        read
    fi
done 