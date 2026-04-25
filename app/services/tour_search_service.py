from collections.abc import Sequence
import json
import logging
import math
from datetime import date
from typing import Any

from app.core.config import Settings
from app.core.exceptions import ValidationServiceError
from app.db.postgres import Database
from app.schemas.tour_search import TourOffer, TourSearchRequest
from app.services.embedding_client import EmbedderClient


class TourSearchService:
    ACCOMMODATION_MAP: dict[tuple[int, int], int] = {
        (1, 0): 1,
        (2, 0): 2,
        (1, 1): 14317,
        (1, 2): 14357,
        (2, 1): 14258,
        (2, 2): 14356,
    }

    def __init__(self, *, db: Database, embedder_client: EmbedderClient, settings: Settings):
        self._db = db
        self._embedder = embedder_client
        self._settings = settings
        self._logger = logging.getLogger("tour_search.service")

    async def search(self, payload: TourSearchRequest) -> list[TourOffer]:
        accommodation_id = self._resolve_accommodation_id(
            num_adults=payload.num_adults,
            num_childs=payload.num_childs,
        )

        children_count, infants_count = self._count_children_groups(
            birthdays=payload.birthdays,
            date_ref=payload.data_min,
        )

        self._logger.info(
            "Executing tour search",
            extra={
                "date_min": str(payload.data_min),
                "date_max": str(payload.data_max),
                "countries_count": len(payload.countries),
                "nights_count": len(payload.num_nights),
                "budget": payload.budget,
                "children_count": children_count,
                "infants_count": infants_count,
                "accommodation_id": accommodation_id,
            },
        )

        rows = await self._db.fetch_ranked_offers(
            date_min=payload.data_min,
            date_max=payload.data_max,
            num_nights=sorted(set(payload.num_nights)),
            countries=[country.strip() for country in payload.countries],
            children_count=children_count,
            infants_count=infants_count,
            accommodation_id=accommodation_id,
            budget=payload.budget,
            limit=self._settings.sql_limit_candidates,
        )

        if not rows:
            self._logger.info("No offers matched SQL filters")
            return []

        query_embedding = await self._embedder.embed_text(payload.query.strip())
        ranked_rows = self._rerank_rows(rows=rows, query_embedding=query_embedding)

        offers: list[TourOffer] = []
        for row in ranked_rows[: self._settings.response_top_k]:
            offers.append(
                TourOffer(
                    teztour_id=int(row["teztour_id"]),
                    price=float(row["price"]),
                    checkin_date=row["checkin_date"],
                    nights=int(row["nights"]),
                )
            )

        self._logger.info(
            "Tour search completed",
            extra={
                "candidate_count": len(rows),
                "result_count": len(offers),
            },
        )

        return offers

    def _resolve_accommodation_id(self, *, num_adults: int, num_childs: int) -> int:
        key = (num_adults, num_childs)
        if key not in self.ACCOMMODATION_MAP:
            self._logger.warning(
                "Unsupported accommodation composition",
                extra={"num_adults": num_adults, "num_childs": num_childs},
            )
            raise ValidationServiceError(
                "Unsupported tourist composition. Supported: "
                "SGL, DBL, SGL+CHD, SGL+CHD2, DBL+CHD, DBL+CHD2"
            )
        return self.ACCOMMODATION_MAP[key]

    def _count_children_groups(self, *, birthdays: list[date], date_ref: date) -> tuple[int, int]:
        children_count = 0
        infants_count = 0

        for birthday in birthdays:
            age = self._age_on_date(birthday=birthday, date_ref=date_ref)
            if age < 0:
                raise ValidationServiceError("Child birthday cannot be in the future")
            if age < 2:
                infants_count += 1
            else:
                children_count += 1

        return children_count, infants_count

    @staticmethod
    def _age_on_date(*, birthday: date, date_ref: date) -> int:
        years = date_ref.year - birthday.year
        if (date_ref.month, date_ref.day) < (birthday.month, birthday.day):
            years -= 1
        return years

    def _rerank_rows(self, *, rows: list[Any], query_embedding: list[float]) -> list[Any]:
        scored_rows: list[tuple[float, Any]] = []
        non_scored_rows: list[Any] = []
        skipped = 0

        for row in rows:
            offer_embedding = self._parse_embedding(row["embedding"])
            if not offer_embedding:
                skipped += 1
                non_scored_rows.append(row)
                continue

            score = self._cosine_similarity(query_embedding, offer_embedding)
            if score is None:
                skipped += 1
                non_scored_rows.append(row)
                continue
            scored_rows.append((score, row))

        scored_rows.sort(key=lambda item: item[0], reverse=True)

        self._logger.debug(
            "Reranking completed",
            extra={
                "input_candidates": len(rows),
                "ranked_candidates": len(scored_rows),
                "skipped_candidates": skipped,
            },
        )

        if not scored_rows:
            self._logger.warning(
                "Reranking produced no scored candidates; falling back to SQL order"
            )
            return list(rows)

        return [row for _, row in scored_rows] + non_scored_rows

    def _parse_embedding(self, raw_embedding: Any) -> list[float]:
        if isinstance(raw_embedding, list):
            return [float(item) for item in raw_embedding]

        if isinstance(raw_embedding, tuple):
            return [float(item) for item in raw_embedding]

        if isinstance(raw_embedding, bytes):
            try:
                raw_embedding = raw_embedding.decode("utf-8")
            except UnicodeDecodeError:
                return []

        if isinstance(raw_embedding, Sequence) and not isinstance(raw_embedding, (str, bytes)):
            try:
                return [float(item) for item in raw_embedding]
            except (TypeError, ValueError):
                return []

        if isinstance(raw_embedding, str):
            value = raw_embedding.strip()
            if not value:
                return []
            if value.startswith("{") and value.endswith("}"):
                value = "[" + value[1:-1] + "]"
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                self._logger.debug("Unable to parse row embedding string")
                return []
            if isinstance(parsed, list):
                return [float(item) for item in parsed]
            return []

        return []

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float | None:
        if len(vec1) != len(vec2):
            return None

        norm1 = math.sqrt(sum(x * x for x in vec1))
        norm2 = math.sqrt(sum(y * y for y in vec2))
        if norm1 == 0 or norm2 == 0:
            return None

        dot = sum(a * b for a, b in zip(vec1, vec2))
        return dot / (norm1 * norm2)
