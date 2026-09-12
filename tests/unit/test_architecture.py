"""
The modules and packages of elicito form a dependency graph without cycles
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).parents[2] / "src" / "elicito"


def _module_name(path: Path) -> str:
    parts = path.relative_to(SRC.parent).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


FILES = {_module_name(path): path for path in sorted(SRC.rglob("*.py"))}
# the package facade imports everything, so it is not part of the graph
MODULES = sorted(name for name in FILES if name != "elicito")


def _is_type_checking(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _imports(module: str) -> set[str]:
    """elicito names that a module imports when it loads, not inside a function"""
    found: set[str] = set()
    body = list(ast.parse(FILES[module].read_text()).body)
    while body:
        node = body.pop()
        if isinstance(node, ast.If) and not _is_type_checking(node):
            body.extend(node.body + node.orelse)
        elif isinstance(node, ast.Try):
            body.extend(node.body + node.orelse + node.finalbody)
            for handler in node.handlers:
                body.extend(handler.body)
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.startswith("elicito"))
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "elicito" or node.module.startswith("elicito."):
                for alias in node.names:
                    # "from elicito.pkg import mod" imports the module pkg.mod
                    submodule = f"{node.module}.{alias.name}"
                    found.add(submodule if submodule in FILES else node.module)
    return found


def _ancestors(name: str) -> list[str]:
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


def _dependencies(module: str) -> set[str]:
    """
    Modules that load when this module loads

    Importing elicito.a.b first runs the package elicito.a. A module inside
    that package is loaded by it, so that implicit edge is not counted; an
    explicit import of the package is.
    """
    own = set(_ancestors(module))
    dependencies: set[str] = set()
    for name in _imports(module):
        implicit = [a for a in _ancestors(name) if a not in own]
        dependencies.update(t for t in [*implicit, name] if t != module)
    return dependencies & set(MODULES)


def _package(module: str) -> str:
    """Top-level part below elicito: the package, or the module itself"""
    return module.split(".")[1]


def _find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    """Return one cycle as a list of nodes, or None"""
    state: dict[str, str] = {}
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        state[node] = "open"
        stack.append(node)
        for nxt in sorted(graph[node]):
            if state.get(nxt) == "open":
                return [*stack[stack.index(nxt) :], nxt]
            if nxt not in state:
                cycle = visit(nxt)
                if cycle is not None:
                    return cycle
        state[node] = "done"
        stack.pop()
        return None

    for node in sorted(graph):
        if node not in state:
            cycle = visit(node)
            if cycle is not None:
                return cycle
    return None


@pytest.mark.parametrize("module", MODULES)
def test_module_does_not_import_the_package(module):
    """a module that imports the package reaches every module, and hides a cycle"""
    assert "elicito" not in _imports(module)


@pytest.mark.parametrize("package", sorted({_package(m) for m in MODULES}))
def test_package_has_a_docstring(package):
    """docs/gen_doc_stubs.py reads the docstring of every module and package"""
    name = f"elicito.{package}"
    assert ast.get_docstring(ast.parse(FILES[name].read_text()))


def test_module_dependencies_have_no_cycle():
    graph = {module: _dependencies(module) for module in MODULES}

    cycle = _find_cycle(graph)

    assert cycle is None, " -> ".join(cycle or [])


def test_packages_do_not_import_each_other_in_a_cycle():
    graph: dict[str, set[str]] = {_package(m): set() for m in MODULES}
    for module in MODULES:
        graph[_package(module)].update(_package(d) for d in _dependencies(module))
    for package, dependencies in graph.items():
        dependencies.discard(package)

    cycle = _find_cycle(graph)

    assert cycle is None, " -> ".join(cycle or [])


def test_find_cycle_reports_a_cycle():
    graph = {"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": set()}

    assert _find_cycle(graph) == ["a", "b", "c", "a"]
