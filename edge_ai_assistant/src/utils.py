def format_bytes(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024 ** 2:
        return f"{n_bytes/1024:.1f} KB"
    elif n_bytes < 1024 ** 3:
        return f"{n_bytes/1024**2:.1f} MB"
    else:
        return f"{n_bytes/1024**3:.1f} GB"
