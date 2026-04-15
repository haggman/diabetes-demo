python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
echo ""
echo "✅ Python env setup complete!"
echo ""
echo "👉 To run the agent test harness in Cloud Shell, use:"
echo ""
echo '   adk web --allow_origins "*"'
echo ""
