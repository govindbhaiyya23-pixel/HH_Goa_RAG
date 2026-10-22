
from google import genai


class GeminiGenerator:

    def __init__(self, api_key):

        self.client = genai.Client(
            api_key=api_key
        )


    def generate(
        self,
        query,
        context
    ):

        prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY
the provided retrieved context.

If the answer cannot be found in the
context, say:

"I don't have enough information in the retrieved context."

Retrieved context:

{context}

Question:

{query}

Give a concise and accurate answer.
"""

        response = self.client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        return response.text
