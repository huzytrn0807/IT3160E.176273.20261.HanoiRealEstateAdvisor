"""Resolve the advertised property group when the title identifies collective housing."""
import re
import unicodedata


COLLECTIVE_HOUSING = "collective_housing"
COLLECTIVE_SUBJECT = (
    r"(?:(?:(?:can ho|can) khu|can nha|can ho|can|nha(?: can ho| chung cu)?|chung cu) )?"
    r"tap the\b(?! duc\b)"
)
COLLECTIVE_TITLE = re.compile(
    rf"^{COLLECTIVE_SUBJECT}|"
    rf"\b(?:ban|chuyen nhuong|sang nhuong) (?:(?:gap|nhanh|chinh chu|duy nhat) )*"
    rf"{COLLECTIVE_SUBJECT}"
)


def property_group_for_title(group: str, title: str | None) -> str:
    """Recognize a collective unit as the subject of the ad, rather than a nearby landmark."""
    if group not in {"apartment", "house"} or not isinstance(title, str):
        return group
    folded = unicodedata.normalize("NFKD", title.casefold().replace("đ", "d"))
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    title_key = " ".join(re.sub(r"[^a-z0-9]+", " ", folded).split())
    return COLLECTIVE_HOUSING if COLLECTIVE_TITLE.search(title_key) else group
