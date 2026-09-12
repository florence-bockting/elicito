"""
The modules of elicito form a dependency graph without cycles
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).parents[2] / "src" / "elicito"
MODULES = sorted(p.stem for p in SRC.glob("*.py") if p.name != "__init__.py")


def _is_type_checking(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _imports(module: str) -> set[str]:
    """elicito names that a module imports when it loads, not inside a function"""
    found: set[str] = set()
    body = list(ast.parse((SRC / f"{module}.py").read_text()).body)
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
            if node.module == "elicito":
                found.update(f"elicito.{a.name}" for a in node.names)
            elif node.module.startswith("elicito."):
                found.add(node.module)
    return found


def _dependencies(module: str) -> set[str]:
    return {name.removeprefix("elicito.") for name in _imports(module)} & set(MODULES)


def _find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    """Return one cycle as a list of modules, or None"""
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


def test_module_dependencies_have_no_cycle():
    graph = {module: _dependencies(module) for module in MODULES}

    cycle = _find_cycle(graph)

    assert cycle is None, " -> ".join(cycle or [])


def test_find_cycle_reports_a_cycle():
    graph = {"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": set()}

    assert _find_cycle(graph) == ["a", "b", "c", "a"]
