from pydantic import BaseModel, Field


class StudentFeatures(BaseModel):
    study_hours: float = Field(..., ge=0, description="Study hours per day")
    attendance: float = Field(..., ge=0, le=100)
    assignment_score: float = Field(..., ge=0, le=10)
    midterm_score: float = Field(..., ge=0, le=10)
    practice_score: float = Field(..., ge=0, le=10)
