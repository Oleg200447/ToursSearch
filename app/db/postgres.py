import logging
from datetime import date
from typing import Sequence

import asyncpg

from app.core.config import Settings
from app.core.exceptions import DataAccessError


class Database:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._pool: asyncpg.Pool | None = None
        self._logger = logging.getLogger("tour_search.db")

    async def connect(self) -> None:
        try:
            self._pool = await asyncpg.create_pool(
                dsn=self._settings.postgres_dsn,
                min_size=self._settings.db_min_pool_size,
                max_size=self._settings.db_max_pool_size,
                command_timeout=self._settings.db_command_timeout_seconds,
            )
            self._logger.info("PostgreSQL pool initialized")
        except Exception as exc:
            self._logger.exception("Unable to initialize PostgreSQL pool")
            raise DataAccessError("Failed to connect to PostgreSQL") from exc

    async def disconnect(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._logger.info("PostgreSQL pool closed")

    async def ping(self) -> None:
        if self._pool is None:
            raise DataAccessError("Database pool is not initialized")
        async with self._pool.acquire() as conn:
            await conn.execute("SELECT 1")

    async def fetch_ranked_offers(
        self,
        *,
        date_min: date,
        date_max: date,
        num_nights: Sequence[int],
        countries: Sequence[str],
        children_count: int,
        infants_count: int,
        accommodation_id: int,
        budget: float,
        limit: int,
    ) -> list[asyncpg.Record]:
        if self._pool is None:
            raise DataAccessError("Database pool is not initialized")

        query = """
        WITH RankedOffers AS (
            SELECT
                h.teztour_id,
                t.price,
                t.checkin_date,
                t.nights,
                h.embedding,
                ROW_NUMBER() OVER(
                    PARTITION BY t.hotel_id
                    ORDER BY t.price ASC
                ) AS rn
            FROM tour_prices_history t
            JOIN hotels h ON t.hotel_id = h.teztour_id
            JOIN resorts r ON h.resort_id = r.resort_id
            JOIN countries c ON r.country_id = c.country_id
            WHERE
                t.checkin_date BETWEEN $1 AND $2
                AND t.nights = ANY($3::int[])
                AND c.country_name = ANY($4::text[])
                AND t.children_count = $5
                AND t.infants_count = $6
                AND t.accommodation_id = $7
        )
        SELECT
            teztour_id,
            price,
            checkin_date,
            nights,
            embedding
        FROM RankedOffers
        WHERE rn = 1
          AND price <= $8
        ORDER BY price DESC
        LIMIT $9
        """

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    query,
                    date_min,
                    date_max,
                    list(num_nights),
                    list(countries),
                    children_count,
                    infants_count,
                    accommodation_id,
                    budget,
                    limit,
                )
                return rows
        except Exception as exc:
            self._logger.exception(
                "Failed to fetch ranked offers from PostgreSQL",
                extra={
                    "date_min": str(date_min),
                    "date_max": str(date_max),
                    "nights_count": len(num_nights),
                    "countries_count": len(countries),
                    "children_count": children_count,
                    "infants_count": infants_count,
                    "accommodation_id": accommodation_id,
                    "budget": budget,
                },
            )
            raise DataAccessError() from exc
