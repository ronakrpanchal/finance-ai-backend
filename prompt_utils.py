from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel
from pathlib import Path

env = Environment(
    loader=FileSystemLoader(Path(__file__).resolve().parent)
)

def prompt_render(prompt_obj: BaseModel) -> str:
    filename = getattr(prompt_obj, "filename", None)
    if not filename:
        raise ValueError("Prompt object must include a 'filename' field.")
    template = env.get_template(filename)
    data = prompt_obj.model_dump()

    return template.render(**data)