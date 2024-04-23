from pathlib import Path


p = Path("./testtreesitter.py")


import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language(), "python")

parser = Parser()
parser.set_language(PY_LANGUAGE)

with open(p, "r", encoding="utf-8") as python_file:
    tree = parser.parse(bytes(python_file.read(), "utf8"))

print(tree.root_node.children)

cursor = tree.walk()

cursor.goto_first_child()

cursor.goto_first_child()

print(cursor.node.type)

