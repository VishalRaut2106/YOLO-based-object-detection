# Setting Up AI Descriptions

The Enhanced Detection System can generate AI-powered descriptions of your snapshots using Google's Gemini API.

## Quick Setup

### Step 1: Get a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click "Get API Key"
4. Create a new API key
5. Copy the key

### Step 2: Set the API Key

**On Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

**On Windows (CMD):**
```cmd
set GEMINI_API_KEY=your_api_key_here
```

**On Linux/Mac:**
```bash
export GEMINI_API_KEY=your_api_key_here
```

### Step 3: Run the System

```bash
python enhanced_detection_system.py -c yolov3.cfg -w yolov3.weights -cl yolov3.txt
```

## How It Works

1. **Press 'S'** to capture a snapshot
2. The system automatically generates an AI description
3. **Press 'D'** to toggle the description panel
4. The description includes:
   - Scene summary
   - List of detected objects
   - Context about the setting

## Without API Key

If you don't set the GEMINI_API_KEY, the system will still work but will generate simple fallback descriptions based only on the detected object labels (no AI analysis).

## Features

- **Automatic Description**: Generated immediately after snapshot
- **Smart Analysis**: Gemini analyzes both the image and detected objects
- **Caching**: Descriptions are cached to avoid redundant API calls
- **Rate Limiting**: Max 2 requests per second to respect API limits
- **Retry Logic**: Automatic retry with exponential backoff on failures
- **Fallback Mode**: Works without API key (basic descriptions)

## Keyboard Controls

- **S** - Capture snapshot + generate AI description
- **D** - Toggle description panel
- **H** - Show help
- **P** - Pause/Resume
- **Q** - Quit

## Example Description

```
Scene Summary: This snapshot shows a workspace with a person working 
on a laptop. The scene appears to be an indoor office or home office 
environment.

Detected Objects:
  - 1 person
  - 1 laptop
  - 1 keyboard
  - 1 mouse

Context: The scene appears to involve human activity in a work setting.
```

## Troubleshooting

### "GEMINI_API_KEY not set" Warning

Set the environment variable as shown in Step 2 above.

### API Errors

- Check your API key is valid
- Ensure you have internet connection
- Check API quota limits at [Google AI Studio](https://aistudio.google.com/)

### Slow Descriptions

- First request may take 5-10 seconds
- Subsequent requests are faster due to caching
- Check your internet speed

## Configuration

Edit `config.json` to customize AI description settings:

```json
{
  "ai_description": {
    "provider": "gemini",
    "model": "gemini-2.0-flash",
    "timeout_seconds": 10,
    "max_retries": 2,
    "cache_enabled": true
  }
}
```

## Cost

- Gemini API has a free tier
- Check current pricing at [Google AI Pricing](https://ai.google.dev/pricing)
- The system uses caching to minimize API calls
