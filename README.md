# Phishing Detection Web Application

A complete, production-ready phishing detection web application built with Flask backend and modern frontend using pre-trained Hugging Face models for URL and email phishing detection.

## Features

- **Email Phishing Detection**: Uses DistilBERT model (`cybersectony/phishing-email-detection-distilbert_v2.4.1`) with 99.58% accuracy
- **URL Phishing Detection**: Uses LinearSVM model (`pirocheto/phishing-url-detection`) with ONNX runtime
- **Modern Responsive UI**: Built with Bootstrap 5, HTML5, CSS3, and JavaScript
- **REST API**: JSON-based API endpoints for integration
- **Real-time Detection**: Instant results with confidence scores

## Models Used

1. **URL Detection Model**:
   - Model: `pirocheto/phishing-url-detection`
   - Type: LinearSVM (ONNX format)
   - Input: URL string
   - Output: Label ("phishing" or "legitimate") with confidence score

2. **Email Detection Model**:
   - Model: `cybersectony/phishing-email-detection-distilbert_v2.4.1`
   - Type: DistilBERT (Text Classification)
   - Input: Subject and body (combined text)
   - Output: Label ("phishing" or "not_phishing") with confidence score

## Project Structure

```
phishing_detector/
├── app/
│   ├── src/
│   │   ├── main.py              # Flask application entry point
│   │   └── static/
│   │       ├── index.html       # Frontend HTML
│   │       ├── style.css        # CSS styles
│   │       ├── script.js        # JavaScript functionality
│   │       └── phishing_warning.jpg  # Warning image
│   ├── venv/                    # Python virtual environment
│   └── requirements.txt         # Python dependencies
└── README.md                    # This file
```

## Setup Instructions for VS Code

### Prerequisites

- Python 3.11 or higher
- VS Code with Python extension
- Git (optional)

### Step-by-Step Setup

1. **Open VS Code and Terminal**
   - Open VS Code
   - Open Terminal (View → Terminal or Ctrl+`)

2. **Navigate to Project Directory**
   ```bash
   cd path/to/phishing_detector/app
   ```

3. **Create and Activate Virtual Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install Additional Dependencies (if needed)**
   ```bash
   pip install flask flask-cors transformers torch onnxruntime joblib huggingface-hub
   ```

6. **Run the Application**
   ```bash
   python src/main.py
   ```

7. **Access the Application**
   - Open your web browser
   - Navigate to: `http://localhost:5000`

### VS Code Configuration

1. **Select Python Interpreter**
   - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)
   - Type "Python: Select Interpreter"
   - Choose the interpreter from your virtual environment: `./venv/Scripts/python.exe` (Windows) or `./venv/bin/python` (macOS/Linux)

2. **Debug Configuration** (Optional)
   - Create `.vscode/launch.json`:
   ```json
   {
       "version": "0.2.0",
       "configurations": [
           {
               "name": "Python: Flask",
               "type": "python",
               "request": "launch",
               "program": "${workspaceFolder}/src/main.py",
               "console": "integratedTerminal",
               "justMyCode": true
           }
       ]
   }
   ```

## API Endpoints

### 1. URL Detection
- **Endpoint**: `POST /predict_url`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "url": "https://example.com"
  }
  ```
- **Response**:
  ```json
  {
    "label": "legitimate",
    "score": 0.95
  }
  ```

### 2. Email Detection
- **Endpoint**: `POST /predict_email`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "subject": "Urgent: Account Verification",
    "body": "Please click here to verify your account..."
  }
  ```
- **Response**:
  ```json
  {
    "label": "phishing",
    "score": 0.99
  }
  ```

## Testing the Application

### Frontend Testing
1. Open `http://localhost:5000` in your browser
2. Test URL Checker:
   - Enter a URL (e.g., `https://www.google.com`)
   - Click "Check URL"
   - View the result with confidence score

3. Test Email Checker:
   - Enter a suspicious subject line
   - Enter email body content
   - Click "Check Email"
   - View the result with confidence score

### API Testing with curl

**Test URL Detection:**
```bash
curl -X POST http://localhost:5000/predict_url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.google.com"}'
```

**Test Email Detection:**
```bash
curl -X POST http://localhost:5000/predict_email \
  -H "Content-Type: application/json" \
  -d '{"subject": "Urgent: Account Verification", "body": "Click here to verify your account immediately"}'
```

## Troubleshooting

### Common Issues

1. **Module Not Found Error**
   - Ensure virtual environment is activated
   - Install missing dependencies: `pip install <module_name>`

2. **URL Detection Model Error**
   - The ONNX model may have locale issues on some systems
   - Email detection will still work perfectly
   - Consider using the pickle version for URL detection if needed

3. **Port Already in Use**
   - Change the port in `main.py`: `app.run(host='0.0.0.0', port=5001)`
   - Or kill the process using the port

4. **Model Download Issues**
   - Ensure internet connection for first-time model downloads
   - Models are cached locally after first download

### Performance Notes

- First run may take longer due to model downloads
- Subsequent runs will be faster as models are cached
- Email detection model (~268MB) loads faster than URL model (~23MB ONNX)

## Development

### Adding New Features
1. Backend changes: Modify `src/main.py`
2. Frontend changes: Update files in `src/static/`
3. Restart the Flask server to see changes

### Model Updates
- Update model names in `main.py`
- Ensure compatibility with the API response format

## Security Considerations

- This is a development server - use a production WSGI server for deployment
- Implement rate limiting for production use
- Add input validation and sanitization
- Consider adding authentication for API access

## License

MIT License - Feel free to use and modify as needed.

