from typing import Type, TypeVar

from beanie import Document

from .task import CutDetectionTaskModel

DocType = TypeVar("DocType", bound=Document)


def gather_documents() -> list[Type[DocType]]:
    return [CutDetectionTaskModel]
