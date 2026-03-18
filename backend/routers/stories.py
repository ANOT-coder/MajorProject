"""
/api/stories  — story keypoint data (for StoryPlayer.jsx)

GET  /api/stories          → list of stories
GET  /api/stories/{id}     → full story with keypoint frames
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.preprocessor import text_to_gloss
from services.vocab        import glosses_to_ids
from services.inference    import run_inference

logger = logging.getLogger(__name__)
router = APIRouter(tags=["stories"])

# ---------------------------------------------------------------------------
# Mock story data — replace with DB queries
# ---------------------------------------------------------------------------
MOCK_STORIES = [
    {
        "id": "1",
        "title": "Good Morning",
        "description": "A simple morning greeting story",
        "level": "beginner",
        "duration": "30s",
        "text": "Good morning. How are you today? I am happy to see you.",
    },
    {
        "id": "2",
        "title": "At the Store",
        "description": "Common phrases used while shopping",
        "level": "beginner",
        "duration": "45s",
        "text": "I want to buy food. Do you have water? How much does this cost? Thank you.",
    },
    {
        "id": "3",
        "title": "My Family",
        "description": "Describing family members",
        "level": "intermediate",
        "duration": "60s",
        "text": "My family is happy. I have a mother and father. My sister is young. We love each other.",
    },
]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class StorySummary(BaseModel):
    id: str
    title: str
    description: str
    level: str
    duration: str


class StoryDetail(StorySummary):
    text: str
    glosses: list[str]
    gloss_ids: list[int]
    frames: list[dict]
    frame_count: int
    fps: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/stories", response_model=list[StorySummary])
async def list_stories():
    return [StorySummary(**{k: s[k] for k in StorySummary.model_fields}, ) for s in MOCK_STORIES]


@router.get("/stories/{story_id}", response_model=StoryDetail)
async def get_story(story_id: str):
    story = next((s for s in MOCK_STORIES if s["id"] == story_id), None)
    if not story:
        raise HTTPException(status_code=404, detail=f"Story '{story_id}' not found.")

    glosses   = text_to_gloss(story["text"])
    gloss_ids, _ = glosses_to_ids(glosses)
    frames    = run_inference(gloss_ids)

    return StoryDetail(
        **{k: story[k] for k in StorySummary.model_fields},
        text=story["text"],
        glosses=glosses,
        gloss_ids=gloss_ids,
        frames=frames,
        frame_count=len(frames),
        fps=25,
    )