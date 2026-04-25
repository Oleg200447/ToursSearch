import logging
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import router as tour_router
from app.core.config import get_settings
from app.core.exceptions import ServiceError
from app.core.logging import configure_logging
from app.db.postgres import Database
from app.services.embedding_client import EmbedderClient
from app.services.tour_search_service import TourSearchService

settings = get_settings()
configure_logging(settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = app.state.logger
    logger.info("Starting application", extra={"app": settings.app_name})

    db = Database(settings)
    await db.connect()

    embedder = EmbedderClient(
        url=settings.embedder_url,
        model=settings.embedder_model,
        timeout_seconds=settings.embedder_timeout_seconds,
    )

    app.state.db = db
    app.state.embedder = embedder
    app.state.tour_search_service = TourSearchService(
        db=db,
        embedder_client=embedder,
        settings=settings,
    )

    logger.info("Application startup completed")
    try:
        yield
    finally:
        await embedder.close()
        await db.disconnect()
        logger.info("Application shutdown completed")


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.state.logger = logging.getLogger("tour_search")


@app.exception_handler(ServiceError)
async def handle_service_error(request: Request, exc: ServiceError) -> JSONResponse:
    logger = request.app.state.logger
    if exc.status_code >= 500:
        logger.error(
            "Service error",
            extra={"status_code": exc.status_code, "error": exc.message},
        )
    else:
        logger.warning(
            "Business validation error",
            extra={"status_code": exc.status_code, "error": exc.message},
        )

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    logger = request.app.state.logger
    logger.warning("Request validation failed", extra={"errors": exc.errors()})
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger = request.app.state.logger
    logger.exception("Unexpected internal error", extra={"path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    logger = request.app.state.logger
    started = perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled exception while processing request",
            extra={"method": request.method, "path": request.url.path},
        )
        raise

    duration_ms = round((perf_counter() - started) * 1000, 2)
    status_code = response.status_code

    message = "Request completed"
    extra = {
        "method": request.method,
        "path": request.url.path,
        "status_code": status_code,
        "duration_ms": duration_ms,
    }

    if status_code >= 500:
        logger.error(message, extra=extra)
    elif status_code >= 400:
        logger.warning(message, extra=extra)
    else:
        logger.info(message, extra=extra)

    return response


@app.get("/health", tags=["health"])
async def health() -> dict:
    return {"status": "ok"}


@app.get("/health/ready", tags=["health"])
async def readiness(request: Request):
    db: Database = request.app.state.db
    try:
        await db.ping()
    except Exception:
        request.app.state.logger.exception("Readiness check failed")
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return {"status": "ready"}


app.include_router(tour_router, prefix=settings.api_prefix)
