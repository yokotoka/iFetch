from typing import Any


def can_read_file(item: Any) -> bool:
    """Check if an item from iCloud Drive is a readable file.

    Args:
        item: An iCloud Drive item to check

    Returns:
        True if the item is a file that can be downloaded
    """
    try:
        return (
            hasattr(item, 'type') and
            item.type not in ('folder', 'app_library') and
            hasattr(item, 'size') and
            hasattr(item, 'open')
        )
    except AttributeError:
        return False
