import logging

import httpx

from app.core.exceptions import ExternalDependencyError


class EmbedderClient:
    def __init__(self, *, url: str, model: str, timeout_seconds: float):
        self._url = url
        self._model = model
        self._logger = logging.getLogger("tour_search.embedder")
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def embed_text(self, text: str) -> list[float]:
        payload = {
            "model": self._model,
            "input": [text],
        }

        try:
            response = await self._client.post(self._url, json=payload)
        except httpx.HTTPError as exc:
            self._logger.exception("Failed to call embedding service")
            raise ExternalDependencyError("Embedding service is unavailable") from exc

        if response.status_code >= 400:
            self._logger.error(
                "Embedding service returned non-success status",
                extra={
                    "status_code": response.status_code,
                    "response_text": response.text[:500],
                },
            )
            raise ExternalDependencyError("Embedding service returned an error")

        try:
            data = response.json()
        except ValueError as exc:
            self._logger.exception("Embedding service returned invalid JSON")
            raise ExternalDependencyError("Embedding service response is invalid") from exc

        embedding = self._extract_embedding(data)
        if not embedding:
            self._logger.error("Empty embedding received from embedding service")
            raise ExternalDependencyError("Embedding vector is empty")

        return embedding

    def _extract_embedding(self, payload: dict) -> list[float]:
        """
        Parse common embedding response shapes.

        Supported formats:
        - {"data": [{"embedding": [...]}]}
        - {"embeddings": [[...]]}
        - {"embedding": [...]} 
        """
        if isinstance(payload.get("data"), list) and payload["data"]:
            first = payload["data"][0]
            if isinstance(first, dict) and isinstance(first.get("embedding"), list):
                return [float(x) for x in first["embedding"]]

        if isinstance(payload.get("embeddings"), list) and payload["embeddings"]:
            first = payload["embeddings"][0]
            if isinstance(first, list):
                return [float(x) for x in first]

        if isinstance(payload.get("embedding"), list):
            return [float(x) for x in payload["embedding"]]

        return []
