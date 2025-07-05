# AI Scribe with ngrok Setup

This guide will help you run your AI Scribe project with ngrok to expose your local server to the internet.

## Prerequisites

1. **Python** (3.8 or higher)
2. **ngrok** - Download from [https://ngrok.com/download](https://ngrok.com/download)
3. **AI Scribe project** - Make sure you're in the project directory

## Installation Steps

### 1. Install ngrok

1. Download ngrok from [https://ngrok.com/download](https://ngrok.com/download)
2. Extract the executable to a folder
3. Add ngrok to your system PATH or place it in a directory that's already in your PATH
4. (Optional) Sign up for a free ngrok account to get a persistent URL

### 2. Verify Installation

Open a terminal/command prompt and run:
```bash
ngrok version
```

You should see the ngrok version information.

## Running the Project

### Option 1: Python Script (Cross-platform)

```bash
python run_with_ngrok.py
```

### Option 2: PowerShell Script (Windows)

```powershell
.\run_with_ngrok.ps1
```

### Option 3: Batch File (Windows)

```cmd
run_with_ngrok.bat
```

## What the Scripts Do

1. **Check Dependencies**: Verify that Python, ngrok, and required packages are installed
2. **Install Missing Packages**: Automatically install FastAPI, uvicorn, and requests if needed
3. **Start FastAPI Server**: Launch your AI Scribe backend on port 8000
4. **Start ngrok Tunnel**: Create a public tunnel to your local server
5. **Display URLs**: Show you all the available endpoints

## Available Endpoints

Once running, you'll have access to:

- **Local API**: `http://localhost:8000`
- **Public API**: `https://[ngrok-url]` (displayed by the script)
- **Frontend**: `https://[ngrok-url]/app`
- **API Documentation**: `https://[ngrok-url]/docs`
- **Health Check**: `https://[ngrok-url]/health`
- **ngrok Dashboard**: `http://localhost:4040` (for monitoring)

## Troubleshooting

### Common Issues

1. **ngrok not found**
   - Make sure ngrok is installed and in your PATH
   - Restart your terminal after adding ngrok to PATH

2. **Port 8000 already in use**
   - Stop any existing services on port 8000
   - Or modify the scripts to use a different port

3. **Python packages missing**
   - The scripts will automatically install required packages
   - If manual installation is needed: `pip install fastapi uvicorn requests`

4. **ngrok tunnel fails to start**
   - Check if you have a free ngrok account (required for newer versions)
   - Verify your internet connection
   - Check ngrok dashboard at `http://localhost:4040` for error details

### Manual Setup (if scripts don't work)

1. **Start the FastAPI server**:
   ```bash
   python api_server.py
   ```

2. **In a new terminal, start ngrok**:
   ```bash
   ngrok http 8000
   ```

3. **Access your application**:
   - Local: `http://localhost:8000`
   - Public: Use the ngrok URL displayed in the terminal

## Security Notes

- The current setup allows CORS from all origins (`allow_origins=["*"]`)
- For production use, configure CORS properly
- ngrok URLs are temporary unless you have a paid account
- Consider using ngrok authentication for additional security

## Stopping the Services

- **Python Script**: Press `Ctrl+C`
- **PowerShell Script**: Press `Ctrl+C`
- **Batch File**: Press any key when prompted

All scripts will automatically clean up processes when stopped.

## Advanced Configuration

### Custom Port

To use a different port, modify the scripts or use:

```bash
python run_with_ngrok.py --port 8080
```

### ngrok Configuration

You can configure ngrok with:
- Custom domains (paid accounts)
- Authentication tokens
- Custom regions

See [ngrok documentation](https://ngrok.com/docs) for more details.

## Support

If you encounter issues:

1. Check the ngrok dashboard at `http://localhost:4040`
2. Review the terminal output for error messages
3. Ensure all prerequisites are properly installed
4. Try the manual setup steps above 