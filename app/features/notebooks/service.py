import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from app.features.notebooks.repository import NotebookRepository
from app.features.notebooks.model import Notebook
from app.features.notebooks.schema import NotebookCreate, NotebookUpdate, NotebookListResponse, NotebookResponse
from app.core.exceptions import NotFoundException
from app.core.logger import logger


class NotebookService:
    def __init__(self, db: AsyncSession):
        self.repo = NotebookRepository(db)

    async def create_notebook(self, data: NotebookCreate, user_id: uuid.UUID) -> Notebook:
        notebook = Notebook(
            user_id=user_id,
            title=data.title,
            description=data.description,
        )
        created = await self.repo.create(notebook)
        logger.info(f"Notebook created: id={created.id}, title='{created.title}', user={user_id}")
        return created

    async def get_notebook(self, notebook_id: uuid.UUID, user_id: uuid.UUID) -> Notebook:
        notebook = await self.repo.get_by_id(notebook_id, user_id)
        if not notebook:
            raise NotFoundException(f"Notebook {notebook_id} not found")
        return notebook

    async def list_notebooks(self, user_id: uuid.UUID, page: int = 1, size: int = 20) -> NotebookListResponse:
        skip = (page - 1) * size
        notebooks = await self.repo.list_by_user(user_id, skip=skip, limit=size)
        total = await self.repo.count_by_user(user_id)
        return NotebookListResponse(
            notebooks=[NotebookResponse.model_validate(n) for n in notebooks],
            total=total,
            page=page,
            size=size,
            has_more=(skip + size) < total,
        )

    async def update_notebook(self, notebook_id: uuid.UUID, data: NotebookUpdate, user_id: uuid.UUID) -> Notebook:
        notebook = await self.get_notebook(notebook_id, user_id)
        if data.title is not None:
            notebook.title = data.title
        if data.description is not None:
            notebook.description = data.description
        return await self.repo.update(notebook)

    async def delete_notebook(self, notebook_id: uuid.UUID, user_id: uuid.UUID) -> None:
        notebook = await self.get_notebook(notebook_id, user_id)
        await self.repo.delete(notebook)
        logger.info(f"Notebook deleted: id={notebook_id}, user={user_id}")