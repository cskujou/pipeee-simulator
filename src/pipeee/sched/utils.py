import contextlib


def remove_all(lst: list, items_to_remove: set) -> list:
    """Remove all items from a list that are in the items_to_remove set.

    For single item removal, uses list.remove() in-place.
    For multiple items, returns a new list via list comprehension.
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    return [item for item in lst if item not in items_to_remove]