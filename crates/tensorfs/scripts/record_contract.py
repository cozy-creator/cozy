"""Proof-only type checks for actual native records, allowing additive fields.

This does not ship in tensorfs and never validates production peer messages.
"""

from __future__ import annotations

import collections.abc
import types
from typing import Any, Literal, NotRequired, TypeAliasType, Union, get_args, get_origin, get_type_hints, is_typeddict


def check(value: object, schema: Any, path: str = "record") -> None:
    if isinstance(schema, TypeAliasType):
        return check(value, schema.__value__, path)
    if is_typeddict(schema):
        assert isinstance(value, dict), (path, value)
        assert schema.__required_keys__ <= value.keys(), (path, schema.__required_keys__, value)
        for name, kind in get_type_hints(schema, include_extras=True).items():
            if name in value:
                check(value[name], kind, f"{path}.{name}")
        return
    origin, args = get_origin(schema), get_args(schema)
    if origin is NotRequired:
        return check(value, args[0], path)
    if origin is Literal:
        assert value in args, (path, value, args)
        return
    if origin in (types.UnionType, Union):
        for kind in args:
            try:
                check(value, kind, path)
                return
            except AssertionError:
                pass
        raise AssertionError((path, value, schema))
    if origin in (list, tuple, collections.abc.Sequence):
        assert isinstance(value, origin), (path, type(value), origin)
        if origin is tuple:
            assert len(value) == len(args), (path, value, args)
            for item, kind in zip(value, args, strict=True):
                check(item, kind, path)
        else:
            for item in value:
                check(item, args[0], path)
        return
    if origin is dict:
        assert isinstance(value, dict), (path, value)
        for key, item in value.items():
            check(key, args[0], path)
            check(item, args[1], f"{path}[{key!r}]")
        return
    if schema is object:
        return
    assert schema in (str, int, bool, float, bytes, type(None)), (path, "unsupported proof annotation", schema)
    assert type(value) is schema, (path, type(value), schema, value)
