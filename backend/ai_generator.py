import httpx
import json
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Ollama's local LLM API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content.

Response Protocol:
- Answer questions using the provided course context
- If the context doesn't contain relevant information, say so clearly
- Do not mention "based on the context" or "based on the search results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        # 120s timeout to handle first-request model loading
        self.http_client = httpx.Client(timeout=120.0)

    def _call_ollama(self, messages: List[Dict]) -> Dict:
        """Make a POST request to the Ollama chat API."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 800
            }
        }

        resp = self.http_client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()

    def generate_response(self, query: str,
                         context: str = "",
                         conversation_history: Optional[str] = None) -> str:
        """
        Generate AI response using provided context.

        Args:
            query: The user's question or request
            context: Retrieved course content to answer from
            conversation_history: Previous messages for context

        Returns:
            Generated response as string
        """

        # Build system content
        system_content = self.SYSTEM_PROMPT
        if conversation_history:
            system_content += f"\n\nPrevious conversation:\n{conversation_history}"

        # Build user message with context
        if context:
            user_content = f"Course context:\n{context}\n\nQuestion: {query}"
        else:
            user_content = query

        # Build messages list
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        # Get response from Ollama
        data = self._call_ollama(messages)
        return data.get("message", {}).get("content", "")
