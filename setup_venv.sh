#!/bin/bash
venv_name="quant_fi_py_venv"
if ! pwd | grep -q "quant_fi_py"; then
    echo "Please run this script from the quant_fi_py directory"
    exit 1
fi
rm -rf "$venv_name"

if command -v python3.14 &>/dev/null; then
    python_cmd="python3.14"
elif command -v python3.13 &>/dev/null; then
    python_cmd="python3.13"
elif command -v python3.12 &>/dev/null; then
    python_cmd="python3.12"
elif command -v python3.11 &>/dev/null; then
    python_cmd="python3.11"
elif command -v python3.10 &>/dev/null; then
    python_cmd="python3.10"
else
    echo "Python 3.10 or later is required but not found. Please install it and try again."
    exit 1
fi

echo "Creating virtual environment using $python_cmd"

# Create a new virtual environment using Python 3.10 or later
$python_cmd -m venv "$venv_name"
# shellcheck disable=SC1090
source "./$venv_name/bin/activate" # On Windows use `whatsapp_transcriber_venv\Scripts\activate`
pip install numpy scipy sympy matplotlib
pip install yfinance
pip install langchain langchain_anthropic langchain_openai langchain_core langchain_community

# pip install gunicorn flask Flask-SQLAlchemy psycopg2-binary pytz python-dateutil
# echo "install fastapi and dependencies"
# pip install fastapi python-multipart uvicorn greenlet typer asyncpg httpx
# pip install twilio requests python-dotenv openai ngrok pyngrok
# pip install pydub audioread firebase_admin tiktoken pycryptodome Pillow
# pip install python-magic-bin fleep
# pip install google-cloud-kms stripe giphy_client matplotlib pytest pytest-asyncio
# pip install deepdiff decouple
# pip install elevenlabs
# pip install prometheus-client # profiling cProfile

# shellcheck disable=SC1090
source "./$venv_name/bin/activate" && pip freeze >requirements.txt # On Windows use `whatsapp_transcriber_venv\Scripts\activate`

echo "Virtual environment created: $VIRTUAL_ENV"

# shellcheck disable=SC2086
echo "# Virtual Environment: $(basename $VIRTUAL_ENV)" >requirements.txt && pip freeze >>requirements.txt

echo "requirements.txt file created with python version $python_cmd"
