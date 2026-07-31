"""Project security and market-index identifiers."""


def normalize_security_symbol(symbol: object) -> str:
    """Convert a source or API code to the project security-symbol format."""
    value = str(symbol).strip()
    if len(value) == 9 and value[2] == ".":
        exchange, code = value.split(".", 1)
        if exchange.lower() in {"sh", "sz", "bj"} and code.isdigit():
            return f"{code}.{exchange.upper()}"
    if len(value) == 9 and value[6] == ".":
        code, exchange = value.split(".", 1)
        if code.isdigit() and exchange.upper() in {"SH", "SZ", "BJ"}:
            return f"{code}.{exchange.upper()}"
    raise ValueError(f"Unsupported security symbol: {symbol!r}")


def normalize_index_code(index_code: object) -> str:
    """Normalize one supported market-index identifier."""
    value = str(index_code).strip()
    if len(value) == 6 and value[0].upper() == "H" and value[1:].isdigit():
        return value.upper()
    if len(value) == 9 and value[2] == ".":
        exchange, code = value.split(".", 1)
        if exchange.lower() in {"sh", "sz", "bj"} and code.isdigit():
            return f"{exchange.lower()}.{code}"
    if len(value) == 9 and value[6] == ".":
        code, exchange = value.split(".", 1)
        if code.isdigit() and exchange.upper() in {"SH", "SZ", "BJ"}:
            return f"{exchange.lower()}.{code}"
    raise ValueError(f"Unsupported market index code: {index_code!r}")
