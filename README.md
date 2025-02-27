# docs-ai-agent

A Python client app that interacts with Azure OpenAI using Semantic Kernel to control lights. Plans to add additional APIs.

## Setup

1. **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd docs-ai-agent
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Create a `.env` file with your Azure OpenAI credentials:**
    ```env
    AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
    AZURE_OPENAI_API_KEY=your-api-key
    AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
    ```

## Usage

Run the application:
```bash
python src/app.py