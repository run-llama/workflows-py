from typing import Any

from pydantic import model_validator

from .base import Base


class ProjectSummary(Base):
    """Summary of a project with deployment count"""

    project_id: str
    project_name: str
    deployment_count: int

    @model_validator(mode="before")
    @classmethod
    def set_default_project_name(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "project_name" not in data or data.get("project_name") is None:
                if "project_id" in data:
                    data["project_name"] = data["project_id"]
        return data


class ProjectsListResponse(Base):
    """Response model for listing projects with deployment counts"""

    projects: list[ProjectSummary]
