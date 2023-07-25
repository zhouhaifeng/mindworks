from jinja2 import Template
from os import path, remove
from shutil import rmtree


def create_content(template: str, notebooks_data: dict, file_name: str):
    """Filling rst template with data

    :param template: jinja template that will be filled with notebook data
    :type template: str
    :param notebooks_data: data structure containing information required to fill template
    :type notebooks_data: dict
    :param file_name: file name
    :type file_name: str
    :returns: Filled template
    :rtype: str

    """
    template = Template(template)
    notebooks_data["notebook"] = "-".join(file_name.split("-")[:-2])
    return template.render(notebooks_data)


def add_content_below(text: str, path: str, line=3) -> bool:
    """Add additional content (like binder button) to existing rst file

    :param text: Text that will be added inside rst file
    :type text: str
    :param path: Path to modified file
    :type path: str
    :param line: Line number that content will be added. Defaults to 3.
    :type line: int
    :returns: Informs about success or failure in modifying file
    :rtype: bool

    """
    try:
        with open(path, "r+", encoding="utf-8") as file:
            current_file = file.readlines()
            current_file[line:line] = text
            file.seek(0)
            file.writelines(current_file)
            return True
    except FileNotFoundError:
        return False


def process_notebook_name(notebook_name: str) -> str:
    """Processes notebook name

    :param notebook_name: Notebook name by default keeps convention:
        [3 digit]-name-with-dashes-with-output.rst,
        example: 001-hello-world-with-output.rst
    :type notebook_name: str
    :returns: Processed notebook name,
        001-hello-world-with-output.rst -> 001. hello world
    :rtype: str

    """
    return (
        notebook_name[:3]
        + "."
        + " ".join(notebook_name[4:].split(".")[0].split("-")[:-2])
    )


def verify_notebook_name(notebook_name: str) -> bool:
    """Verification based on notebook name

    :param notebook_name: Notebook name by default keeps convention:
        [3 digit]-name-with-dashes-with-output.rst,
        example: 001-hello-world-with-output.rst
    :type notebook_name: str
    :returns: Return if notebook meets requirements
    :rtype: bool

    """
    return notebook_name[:3].isdigit() and notebook_name[-4:] == ".rst"


def split_notebooks_into_sections(notebooks: list) -> list:
    series = [list() for _ in range(5)]
    for notebook in notebooks:
        try:
            series[int(notebook.name[0])].append(notebook)
        except IndexError:
            pass
    return series