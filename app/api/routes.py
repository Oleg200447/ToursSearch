from fastapi import APIRouter, Request

from app.schemas.tour_search import TourSearchRequest, TourSearchResponse
from app.services.tour_search_service import TourSearchService

router = APIRouter(tags=["tour-search"])


@router.post("/tours/search", response_model=TourSearchResponse)
async def search_tours(payload: TourSearchRequest, request: Request) -> TourSearchResponse:
    service: TourSearchService = request.app.state.tour_search_service
    offers = await service.search(payload)
    return TourSearchResponse(offers=offers)
