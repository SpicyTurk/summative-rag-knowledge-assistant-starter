from flask import Flask, jsonify, request
from flask_cors import CORS

from config import Config
from rag_service import answer_question

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": Config.CLIENT_ORIGIN}})


@app.get("/api/health")
def health():
    """Confirm that the backend is running."""
    return jsonify(
        {
            "status": "ok",
            "message": "Backend is running.",
        }
    ), 200


@app.post("/api/ask")
def ask_question():

    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required."}), 400

    try:
        result = answer_question(question)
        return jsonify(result), 200
    
    except requests.exceptions.ConnectionError:

        return jsonify(
            {
                "error": "Could not reach the model service. Is Ollama running?",
                "sources": [],
            }
        ), 503
    except Exception as error:

        app.logger.exception("Error while answering question")
        return jsonify(
            {
                "error": f"Something went wrong while answering: {error}",
                "sources": [],
            }
        ), 500

if __name__ == "__main__":
    app.run(debug=True, port=5555)
