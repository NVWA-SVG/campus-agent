from .gateway import CampusBusinessGateway
from .models import CampusServiceStatus, utc_now_iso, validate_query

_SERVICES = {
    "campus_card": (
        "校园卡服务中心",
        "行政楼一楼 103",
        "08:30-17:30",
        "open",
        6,
        12,
    ),
    "registrar": (
        "教务服务中心",
        "行政楼二楼 205",
        "08:30-17:00",
        "busy",
        18,
        25,
    ),
    "library": ("图书馆服务台", "图书馆一楼", "08:00-22:00", "open", 2, 5),
    "student_affairs": (
        "学生事务中心",
        "学生服务楼 101",
        "09:00-17:00",
        "closed",
        0,
        0,
    ),
}


class MockCampusBusinessGateway(CampusBusinessGateway):
    def query_service_status(
        self, service_code: str, campus: str
    ) -> CampusServiceStatus:
        service_code, campus = validate_query(service_code, campus)
        name, location, hours, status, queue, wait = _SERVICES[service_code]
        now = utc_now_iso()
        return CampusServiceStatus(
            "1.0",
            "mock",
            service_code,
            name,
            campus,
            status,
            location,
            hours,
            now,
            now,
            queue,
            wait,
            request_id="mock-demo",
        )
