"""
Some data classes for the rest of the scripts in this folder.
"""

from dataclasses import dataclass


class Dictionariable:
    def to_dictionary(self):
        raise NotImplementedError()


@dataclass
class ComponentImport(Dictionariable):
    """Dataclass representing an `from X import Y` statement."""

    library: str = None
    package: str = None
    component: str = None
    component_alias: str = None

    def get_used_alias(self) -> str:
        if self.component_alias:
            return self.component_alias
        else:
            return self.component

    def __hash__(self) -> int:
        return hash((self.library, self.package, self.component, self.component_alias))

    def to_dictionary(self):
        return {
            "library": self.library,
            "package": self.package,
            "component": self.component,
            "component_alias": self.component_alias,
        }


@dataclass
class CompositeComponentImport(ComponentImport, Dictionariable):
    def __init__(
        self, component_a: ComponentImport, component_b: ComponentImport
    ) -> None:
        if component_a.get_used_alias() != component_b.get_used_alias():
            raise ValueError("The aliases must match.")

        self.inner_component_imports = []
        for ele in [component_a, component_b]:
            if isinstance(ele, CompositeComponentImport):
                self.inner_component_imports.extend(ele.inner_component_imports)
            else:
                self.inner_component_imports.append(ele)

        composite_library = f"{component_a.library},{component_b.library}"
        composite_package = f"{component_a.package},{component_b.package}"
        if component_a.component == component_b.component:
            composite_component = component_a.component
        else:
            composite_component = f"{component_a.component},{component_b.component}"
        super().__init__(
            composite_library,
            composite_package,
            composite_component,
            component_a.component_alias,
        )

    def to_dictionary(self):
        return {
            "library": self.library,
            "package": self.package,
            "component": self.component,
            "component_alias": self.component_alias,
        }

    def __hash__(self) -> int:
        return super().__hash__()


@dataclass
class PackageImport(Dictionariable):
    """Dataclass representing `import X as Y` statement, or `import X`
    if no alias is used."""

    library: str = None
    package: str = None
    alias: str | None = None

    def get_used_alias(self) -> str:
        if self.alias:
            return self.alias
        else:
            return self.package

    def __hash__(self) -> int:
        return hash((self.library, self.package, self.alias))

    def to_dictionary(self):
        return {
            "library": self.library,
            "package": self.package,
            "alias": self.alias,
        }


def get_library_from_package(package_name: str) -> str:
    return package_name.split(".")[0]
